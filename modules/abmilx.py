import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_mil import Mlp,sdpa

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight) # Initialize linear layer weights using Xavier normal distribution
            if m.bias is not None:
                m.bias.data.zero_() # Initialize bias to zero if it exists
        elif isinstance(m,nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0) # Initialize LayerNorm bias to zero
            nn.init.constant_(m.weight, 1.0) # Initialize LayerNorm weight to one
    
    for m in module.modules():
        if hasattr(m,'init_weights'):
            m.init_weights() # Call custom init_weights if defined in a submodule

class MLPAttn(nn.Module):
    def __init__(self,L=128,D=None,norm=None,bias=False,norm_2=False,dp=0.,k=1,act='gelu',gated=False,dropout=True):
        super(MLPAttn, self).__init__()

        D = D or L # Set D to L if D is not provided
        
        self.gated = gated # Flag for using gated attention

        if self.gated:
            self.attention_a = [
            nn.Linear(L, D,bias=bias), # Linear transformation for attention (part a)
            ]
            if act == 'gelu': 
                self.attention_a += [nn.GELU()] # GELU activation
            elif act == 'relu':
                self.attention_a += [nn.ReLU()] # ReLU activation
            elif act == 'tanh':
                self.attention_a += [nn.Tanh()] # Tanh activation
            elif act == 'swish':
                self.attention_a += [nn.SiLU()] # SiLU (Swish) activation

            self.attention_b = [nn.Linear(L, D,bias=bias), # Linear transformation for attention (part b - gate)
                                nn.Sigmoid()] # Sigmoid activation for gate

            if dropout:
                self.attention_a += [nn.Dropout(0.25)] # Dropout for attention part a
                self.attention_b += [nn.Dropout(0.25)] # Dropout for attention part b (gate)

            self.attention_a = nn.Sequential(*self.attention_a)
            self.attention_b = nn.Sequential(*self.attention_b)

            self.attention_c = nn.Linear(D, k,bias=bias) # Final linear transformation for attention output
        else:
            self.fc1 = nn.Linear(L, D,bias=bias) # First linear layer for non-gated attention

        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'swish':
            self.act = nn.SiLU()

        self.norm_type = norm # Type of normalization to use
        self.dp = nn.Dropout(dp) if dp else nn.Identity() # Dropout layer
        if norm == 'bn':
            self.norm1 = nn.BatchNorm1d(D) # Batch normalization for the first stage
            if norm_2:
                self.norm2 = nn.BatchNorm1d(1) # Batch normalization for the second stage (if enabled)
            else:
                self.norm2 = nn.Identity()
        elif norm == 'ln':
            self.norm1 = nn.LayerNorm(D) # Layer normalization for the first stage
            self.norm2 = nn.Identity() # No second stage normalization for LayerNorm
        else:
            self.norm1 = nn.Identity() # No normalization
            self.norm2 = nn.Identity()
        
        if not self.gated:
            self.fc2 = nn.Linear(D, k,bias=bias) # Second linear layer for non-gated attention
    
    def forward(self,x):
        #assert len(x.shape) == 4
        if len(x.shape) == 3:
            B, N, C = x.shape # Batch size, Number of instances, Channels
            attn = x
            x_shape_3 = True
        else:
            B, H, N, C = x.shape # Batch size, Heads, Number of instances, Channels
            attn = x.reshape(B*H,N,C) # Reshape for multi-head attention processing
            x_shape_3 = False

        if self.gated:
            attn_a = self.attention_a(attn) # Apply first part of gated attention
            attn_b = self.attention_b(attn) # Apply gate part of gated attention
            attn = attn_a.mul(attn_b) # Element-wise multiplication for gating
            attn = self.attention_c(attn) # Final linear transformation
        else:
            attn = self.fc1(attn) # Apply first linear layer
            if self.norm_type == 'bn':
                attn = self.norm1(attn.transpose(-1,-2)).transpose(-1,-2) # Apply batch norm
            else:
                attn = self.norm1(attn) # Apply layer norm or identity
            attn = self.act(attn) # Apply activation function
            attn = self.dp(attn) # Apply dropout

            attn = self.fc2(attn) # Apply second linear layer
            if self.norm_type == 'bn':
                attn = self.norm2(attn.transpose(-1,-2)).transpose(-1,-2) # Apply second batch norm (if enabled)
            else:
                attn = self.norm2(attn) # Apply identity

        if x_shape_3:
            attn = attn.transpose(-1,-2).unsqueeze(-1) # Reshape attention scores for 3D input
        else:
            attn = attn.reshape(B, H, N, 1) # Reshape attention scores for 4D input (multi-head)
        return attn        

class AttnPlus(nn.Module):
    def __init__(self,dim=128,attn_dropout=0.,norm=True,embed=True,sdpa_type='torch',head=8,shortcut=True,v_embed=True,pad_v=False):
        super(AttnPlus, self).__init__()
        self.scale = dim ** -0.5 # Scaling factor for attention
        self.attn_drop = nn.Dropout(attn_dropout) # Dropout for attention scores
        self.embed = embed # Flag for embedding Q, K
        self.sdpa_type = sdpa_type # Type of scaled dot-product attention implementation
        self.shortcut = shortcut # Flag for using shortcut connection
        self.pad_v = pad_v # Flag for padding V tensor
        self.head = head # Number of attention heads
        self.dim = dim # Dimension of the input features

        if embed:
            self.qk = nn.Linear(dim, dim * 2, bias = False) # Linear layer for Q and K projections
            self.v = nn.Linear(1, 1, bias = False) if v_embed else nn.Identity() # Linear layer for V projection (if v_embed is True)
        if norm:
            self.norm_x = nn.LayerNorm(dim) # Layer normalization for input X
        else:
            self.norm_x = nn.Identity()

    def forward(self,x,A):
        if len(x.shape) == 3:
            B, N, _ = x.shape # Batch size, Number of instances, Dimension
            # Project and reshape Q, K for multi-head attention
            qk = self.qk(self.norm_x(x)).reshape(B, N, 2 ,self.head,self.dim // self.head).permute(2, 0, 3, 1, 4)
            q,k = qk.unbind(0) # Separate Q and K
        # B H N D
        else:
            B, H, N, D = x.shape # Batch size, Heads, Number of instances, Dimension per head
            # Project and reshape Q, K for multi-head attention (when input is already shaped for heads)
            qk = self.qk(self.norm_x(x)).reshape(B, self.head, N ,2 ,self.dim)
            q,k = qk.unbind(-2) # Separate Q and K

        v = self.v(A) # Project V (attention scores from previous module)

        if self.sdpa_type not in ('torch_math','torch') or self.pad_v:
            v = F.pad(v, (0, q.shape[-1] - v.size(-1))) # Pad V if necessary for certain SDPA types
        
        # Scaled Dot-Product Attention
        A_plus = sdpa(
            q,k,v,
            attn_drop=self.attn_drop,
            scale=self.scale,
            training=self.training,
            sdpa_type=self.sdpa_type
        ).transpose(1,2)

        if self.sdpa_type not in ('torch_math','torch') or self.pad_v:
            A_plus = A_plus[:,:,:,0].unsqueeze(-1) # Adjust shape if V was padded
        
        if self.shortcut:
            A = A + A_plus # Add shortcut connection
        else:
            A = A_plus # No shortcut
        
        return A
        
class DAttentionX(nn.Module):
    def __init__(self,input_dim,n_classes,dropout=0.25,act='gelu',mil_norm=None,mil_bias=False,inner_dim=512,n_heads=8,proj_drop=0.,D=None,attn_type='mlp',attn_bias=False,attn_plus=True,ffn=False,sdpa_type='torch',pad_v=False,attn_plus_embed_new=False,**kwargs):
        super(DAttentionX, self).__init__()
        self.head_dim = inner_dim // n_heads # Dimension per attention head
        self.L = inner_dim if attn_plus_embed_new else self.head_dim # Input dimension for MLPAttn
        if D is None:
            self.D = self.L # Intermediate dimension for MLPAttn
        else:
            if D > 10.: # If D is a large number, treat as absolute dimension
                self.D = int(D) if attn_plus_embed_new else int(D) // n_heads  
            else: # If D is small, treat as a ratio to L
                self.D = int(self.L // D)

        self.K = 1 # Output dimension of MLPAttn (before head aggregation if attn_plus_embed_new is false)
        self.feature = [] # List to store feature extractor layers
        self.mil_norm = mil_norm # Type of normalization for MIL
        self.n_heads = n_heads # Number of attention heads
        self.attn_plus = attn_plus # Flag to use AttnPlus module
        self.attn_plus_embed_new = attn_plus_embed_new # Flag for new embedding strategy in AttnPlus

        if attn_plus:
            # Initialize AttnPlus module
            self.attn_plus_fn = AttnPlus(inner_dim if attn_plus_embed_new else self.head_dim,sdpa_type=sdpa_type,head=n_heads,pad_v=pad_v)

        if ffn:
            self.norm_ffn = nn.LayerNorm(inner_dim) if not mil_norm else nn.Identity() # LayerNorm before FFN
            self.mlp = Mlp( # Feed-Forward Network
            in_features=inner_dim,
            hidden_features=int(inner_dim * 4.),
            act_layer=nn.GELU,
            )
        self.ffn = ffn # Flag to use FFN

        if mil_norm == 'bn':
            self.norm = nn.BatchNorm1d(input_dim) # Batch normalization for input
            self.norm1 = nn.BatchNorm1d(self.L*self.K) # Batch normalization after attention
        elif mil_norm == 'ln':
            self.feature += [nn.LayerNorm(input_dim,bias=mil_bias)] # Layer normalization for input
            self.norm1 = nn.LayerNorm(inner_dim,bias=mil_bias) # Layer normalization after attention

        else:
            self.norm1 = self.norm = nn.Identity() # No normalization
        
        self.feature += [nn.Linear(input_dim, inner_dim,bias=mil_bias)] # Linear layer for feature extraction
        if act.lower() == 'gelu':
            self.feature += [nn.GELU()] # GELU activation
        elif act.lower() == 'relu':
            self.feature += [nn.ReLU()] # ReLU activation
        if dropout:
            self.feature += [nn.Dropout(dropout)] # Dropout layer

        self.feature = nn.Sequential(*self.feature) if len(self.feature) > 0 else nn.Identity()

        if attn_type == 'mlp':
            # Initialize MLPAttn for standard MLP-based attention
            self.attention = MLPAttn(self.L,self.D,bias=attn_bias,
            k=n_heads if attn_plus_embed_new else 1)
        ## only for ablation study
        elif attn_type == 'mlp_gated':
            # Initialize MLPAttn for gated MLP-based attention
            self.attention = MLPAttn(self.L,self.D,bias=attn_bias,
            k=n_heads if attn_plus_embed_new else 1,
            gated=True)

        self.proj = nn.Linear(inner_dim, inner_dim,bias=mil_bias) # Projection layer after attention
        self.proj_drop = nn.Dropout(proj_drop) # Dropout for projection layer

        self.classifier = nn.Linear(inner_dim, n_classes,bias=mil_bias) # Classifier layer

        self.apply(initialize_weights) # Apply weight initialization
    
    def forward(self, x, return_attn=False,return_act=False,return_img_feat=False,**kwargs):
        if len(x.size()) == 2:
            x.unsqueeze_(0) # Add batch dimension if missing

        B, N, _ = x.shape # Batch size, Number of instances, Input dimension

        if self.mil_norm == 'bn':
            x = torch.transpose(x, -1, -2) # Transpose for BatchNorm1d
            x = self.norm(x) # Apply batch normalization
            x = torch.transpose(x, -1, -2) # Transpose back

        x = self.feature(x) # Extract features

        _,_,C = x.shape # Get the channel dimension after feature extraction (inner_dim)
        act = x.clone() # Clone for returning activations if needed

        if self.attn_plus_embed_new:
            A = self.attention(x)   # B N D (Attention scores from MLPAttn)
        else:
            x = x.reshape(B,N,self.n_heads,self.head_dim).transpose(1,2) # B H N D (Reshape for multi-head)
            A = self.attention(x)   # B H N K (Attention scores from MLPAttn per head)

        if self.attn_plus:
            A = self.attn_plus_fn(x,A) # Apply AttnPlus to refine attention scores
        A = torch.transpose(A, -1, -2)  # B H K N (Transpose for softmax)
        A = F.softmax(A, dim=-1)  # softmax over N (Normalize attention scores)

        if self.attn_plus_embed_new:
            # Reshape x for multi-head if new embedding strategy was used (x wasn't reshaped before AttnPlus)
            x = x.reshape(B,N,self.n_heads,self.head_dim).transpose(1,2) # B H N D

        # Weighted sum of features based on attention scores
        x = torch.einsum('b h k n, b h n d -> b h k d', A,x).squeeze(1) # B H D (or B K D if K > 1)

        x = x.reshape(B, C) # Reshape to (Batch size, inner_dim)

        x = self.proj(x) # Apply projection layer
        x = self.proj_drop(x) # Apply dropout

        if self.ffn:
            x = x + self.mlp(self.norm_ffn(x)) # Apply FFN with residual connection

        x = self.norm1(x) # Apply normalization (LayerNorm or BatchNorm)
        _logits = self.classifier(x) # Get classification logits

        if return_img_feat:
            _logits = [_logits,x] # Return logits and image features if requested

        if return_attn:
            output = []
            output.append(_logits)
            output.append(A.squeeze(-2)) # Return attention scores (squeezing the K_out dimension if it's 1)
            if return_act:
                output.append(act.squeeze(1)) # Return activations if requested
            return output
        else:   
            return _logits # Return only logits



