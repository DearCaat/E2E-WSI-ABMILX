import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_mil import Mlp,sdpa

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    for m in module.modules():
        if hasattr(m,'init_weights'):
            m.init_weights()

class MLPAttn(nn.Module):
    def __init__(self,L=128,D=None,norm=None,bias=False,norm_2=False,dp=0.,k=1,act='gelu',gated=False,dropout=True):
        super(MLPAttn, self).__init__()

        D = D or L
        
        self.gated = gated

        if self.gated:
            self.attention_a = [
            nn.Linear(L, D,bias=bias),
            ]
            if act == 'gelu': 
                self.attention_a += [nn.GELU()]
            elif act == 'relu':
                self.attention_a += [nn.ReLU()]
            elif act == 'tanh':
                self.attention_a += [nn.Tanh()]
            elif act == 'swish':
                self.attention_a += [nn.SiLU()]

            self.attention_b = [nn.Linear(L, D,bias=bias),
                                nn.Sigmoid()]

            if dropout:
                self.attention_a += [nn.Dropout(0.25)]
                self.attention_b += [nn.Dropout(0.25)]

            self.attention_a = nn.Sequential(*self.attention_a)
            self.attention_b = nn.Sequential(*self.attention_b)

            self.attention_c = nn.Linear(D, k,bias=bias)
        else:
            self.fc1 = nn.Linear(L, D,bias=bias)

        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'swish':
            self.act = nn.SiLU()

        self.norm_type = norm
        self.dp = nn.Dropout(dp) if dp else nn.Identity()
        if norm == 'bn':
            self.norm1 = nn.BatchNorm1d(D)
            if norm_2:
                self.norm2 = nn.BatchNorm1d(1)
            else:
                self.norm2 = nn.Identity()
        elif norm == 'ln':
            self.norm1 = nn.LayerNorm(D)
            self.norm2 = nn.Identity()
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        if not self.gated:
            self.fc2 = nn.Linear(D, k,bias=bias)
    
    def forward(self,x):
        #assert len(x.shape) == 4
        if len(x.shape) == 3:
            B, N, C = x.shape
            attn = x
            x_shape_3 = True
        else:
            B, H, N, C = x.shape
            attn = x.reshape(B*H,N,C)
            x_shape_3 = False

        if self.gated:
            attn = self.attention_a(attn)
            b = self.attention_b(attn)
            attn = attn.mul(b)
            attn = self.attention_c(attn)
        else:
            attn = self.fc1(attn)
            if self.norm_type == 'bn':
                attn = self.norm1(attn.transpose(-1,-2)).transpose(-1,-2)
            else:
                attn = self.norm1(attn)
            attn = self.act(attn)
            attn = self.dp(attn)

            attn = self.fc2(attn)
            if self.norm_type == 'bn':
                attn = self.norm2(attn.transpose(-1,-2)).transpose(-1,-2)
            else:
                attn = self.norm2(attn)

        if x_shape_3:
            attn = attn.transpose(-1,-2).unsqueeze(-1)
        else:
            attn = attn.reshape(B, H, N, 1)
        return attn        

class AttnPlus(nn.Module):
    def __init__(self,dim=128,attn_dropout=0.,norm=True,embed=True,sdpa_type='torch',head=8,shortcut=True,v_embed=True,pad_v=False):
        super(AttnPlus, self).__init__()
        self.scale = dim ** -0.5
        self.attn_drop = nn.Dropout(attn_dropout)
        self.embed = embed
        self.sdpa_type = sdpa_type
        self.shortcut = shortcut
        self.rope = None
        self.alibi = False
        self.pad_v = pad_v
        self.head = head
        self.dim = dim

        if embed:
            self.qk = nn.Linear(dim, dim * 2, bias = False)
            self.v = nn.Linear(1, 1, bias = False) if v_embed else nn.Identity()
        if norm:
            self.norm_x = nn.LayerNorm(dim)
        else:
            self.norm_x = nn.Identity()

    def forward(self,x,A):
        if len(x.shape) == 3:
            B, N, _ = x.shape
            qk = self.qk(self.norm_x(x)).reshape(B, N, 2 ,self.head,self.dim // self.head).permute(2, 0, 3, 1, 4)
            q,k = qk.unbind(0)
        # B H N D
        else:
            B, H, N, D = x.shape
            qk = self.qk(self.norm_x(x)).reshape(B, self.head, N ,2 ,self.dim)
            q,k = qk.unbind(-2)

        v = self.v(A)

        if self.sdpa_type not in ('torch_math','torch') or self.pad_v:
            v = F.pad(v, (0, q.shape[-1] - v.size(-1)))
        A_plus = sdpa(
            q,k,v,
            attn_drop=self.attn_drop,
            scale=self.scale,
            training=self.training,
            sdpa_type=self.sdpa_type
        ).transpose(1,2)

        if self.sdpa_type not in ('torch_math','torch') or self.pad_v:
            A_plus = A_plus[:,:,:,0].unsqueeze(-1)
        
        if self.shortcut:
            A = A + A_plus
        
        return A
        
class DAttentionX(nn.Module):
    def __init__(self,input_dim,n_classes,dropout=0.25,act='gelu',mil_norm=None,mil_bias=False,inner_dim=512,n_heads=8,proj_drop=0.,D=None,attn_type='mlp',attn_bias=False,attn_plus=True,ffn=False,sdpa_type='torch',pad_v=False,attn_plus_embed_new=False,**kwargs):
        super(DAttentionX, self).__init__()
        self.head_dim = inner_dim // n_heads
        self.L = inner_dim if attn_plus_embed_new else self.head_dim
        if D is None:
            self.D = self.L
        else:
            if D > 10.:
                self.D = int(D) if attn_plus_embed_new else int(D) // n_heads  
            else:
                self.D = int(self.L // D)

        self.K = 1
        self.feature = []
        self.mil_norm = mil_norm
        self.n_heads = n_heads
        self.attn_plus = attn_plus
        self.attn_plus_embed_new = attn_plus_embed_new

        if attn_plus:
            self.attn_plus_fn = AttnPlus(inner_dim if attn_plus_embed_new else self.head_dim,sdpa_type=sdpa_type,head=n_heads,pad_v=pad_v)

        if ffn:
            self.norm_ffn = nn.LayerNorm(inner_dim) if not mil_norm else nn.Identity()
            self.mlp = Mlp(
            in_features=inner_dim,
            hidden_features=int(inner_dim * 4.),
            act_layer=nn.GELU,
            )
        self.ffn = ffn

        if mil_norm == 'bn':
            self.norm = nn.BatchNorm1d(input_dim)
            self.norm1 = nn.BatchNorm1d(self.L*self.K)
        elif mil_norm == 'ln':
            self.feature += [nn.LayerNorm(input_dim,bias=mil_bias)]
            self.norm1 = nn.LayerNorm(inner_dim,bias=mil_bias)

        else:
            self.norm1 = self.norm = nn.Identity()
        
        self.feature += [nn.Linear(input_dim, inner_dim,bias=mil_bias)]
        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        elif act.lower() == 'relu':
            self.feature += [nn.ReLU()]
        if dropout:
            self.feature += [nn.Dropout(dropout)]

        self.feature = nn.Sequential(*self.feature) if len(self.feature) > 0 else nn.Identity()

        if attn_type == 'mlp':
            self.attention = MLPAttn(self.L,self.D,bias=attn_bias,
            k=n_heads if attn_plus_embed_new else 1)
        ## only for ablation study
        elif attn_type == 'mlp_gated':
            self.attention = MLPAttn(self.L,self.D,bias=attn_bias,
            k=n_heads if attn_plus_embed_new else 1,
            gated=True)
        elif attn_type == 'swiglu':
            raise NotImplementedError

        self.proj = nn.Linear(inner_dim, inner_dim,bias=mil_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.classifier = nn.Linear(inner_dim, n_classes,bias=mil_bias)

        self.apply(initialize_weights)
    
    def forward(self, x, return_attn=False,return_act=False,return_img_feat=False,**kwargs):
        if len(x.size()) == 2:
            x.unsqueeze_(0)

        B, N, _ = x.shape

        if self.mil_norm == 'bn':
            x = torch.transpose(x, -1, -2)
            x = self.norm(x)
            x = torch.transpose(x, -1, -2)

        x = self.feature(x)

        _,_,C = x.shape
        act = x.clone()

        if self.attn_plus_embed_new:
            A = self.attention(x)   # B N D
        else:
            x = x.reshape(B,N,self.n_heads,self.head_dim).transpose(1,2) # B H N D
            A = self.attention(x)   # B H N K

        if self.attn_plus:
            A = self.attn_plus_fn(x,A)
        A = torch.transpose(A, -1, -2)  # B H K N
        A = F.softmax(A, dim=-1)  # softmax over N

        if self.attn_plus_embed_new:
            x = x.reshape(B,N,self.n_heads,self.head_dim).transpose(1,2) # B H N D

        x = torch.einsum('b h k n, b h n d -> b h k d', A,x).squeeze(1) # B H D

        x = x.reshape(B, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.ffn:
            x = x + self.mlp(self.norm_ffn(x))

        x = self.norm1(x)
        _logits = self.classifier(x)

        if return_img_feat:
            _logits = [_logits,x]

        if return_attn:
            output = []
            output.append(_logits)
            output.append(A.squeeze(-2))
            if return_act:
                output.append(act.squeeze(1))
            return output
        else:   
            return _logits



