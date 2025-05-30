import torch
import torch.nn as nn
import numpy as np
from timm.layers import use_fused_attn
import torch.nn.functional as F
from modules.emb_position import *
from einops import repeat
from torch.nn.attention import SDPBackend, sdpa_kernel
from contextlib import suppress
from functools import partial

from .nystrom_attention import NystromSDPA

try:
    from flash_attn import flash_attn_qkvpacked_func,flash_attn_varlen_func,flash_attn_func
    from .utils import *
    import xformers.ops as xops
except:
    #print(f"Not installed Flash Attention")
    pass

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = bias
        drop_probs = drop
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class SDPA(nn.Module):
    def __init__(self,sdpa_type='torch',head=8) -> None:
        super().__init__()
        if sdpa_type == 'ntrans':
            self.sdpa_model = NystromSDPA(heads=head,residual=False)

    def forward(self,q=None, k=None ,v=None, attn_drop=None, attn_mask=None,scale=None,training=True,sdpa_type='torch',deterministic=True):
        # q k v  B H N D
        B,H,_,_ = q.shape
        if sdpa_type == 'math':
            attn = (q * scale) @ k.transpose(-2, -1)
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn = attn.masked_fill(~attn_mask, -torch.finfo(attn.dtype).max)
                else:
                    attn += attn_mask
            attn = attn.softmax(dim=-1)
            attn = attn_drop(attn)
            x = attn @ v
            return x.transpose(1, 2)
        elif sdpa_type == 'memo_effi':
            if attn_mask is not None:
                L, S = q.size(1), k.size(1)
                if S % 8 != 0:
                    # 创建一个更大的张量并进行切片以确保内存对齐
                    attn_bias = torch.zeros(B, 1, L, (S // 8 + 1) * 8, dtype=q.dtype, device=q.device)[:, :, :, :S]
                else:
                    attn_bias = torch.zeros(B, 1, L, S, dtype=q.dtype, device=q.device)
                if attn_mask.dtype == torch.bool:
                    attn_bias.masked_fill_(~attn_mask, -torch.finfo(q.dtype).max)
                    attn_bias = repeat(attn_bias,'b 1 n l -> b h n l',h=H)
                else:
                    attn_bias += attn_mask
            else:
                attn_bias = None
            x = xops.memory_efficient_attention(
                    q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
                    attn_bias=attn_bias,
                    p=attn_drop.p if training else 0.,
                )
            return x
        elif sdpa_type == 'flash':
            assert attn_mask is None
            # flash-attn
            x = flash_attn_func(
                q.transpose(1,2),
                k.transpose(1,2),
                v.transpose(1,2),
                dropout_p = attn_drop.p if training else 0.,
                deterministic=deterministic if training else False
            )
            return x
        elif sdpa_type == 'ntrans':
            return self.sdpa_model(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))
        elif sdpa_type == 'torch':
            #if attn_mask is not None or not sdpa_type == 'flash':
            with sdpa_kernel(SDPBackend.MATH):
                x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p = attn_drop.p if training else 0.,
                )
            return x.transpose(1, 2)
        else:
            raise NotImplementedError

def sdpa(q=None, k=None, v=None, attn_drop=None, attn_mask=None,scale=None,training=True,sdpa_type='torch',deterministic=True):
    # q k v  B H N D
        B,H,_,_ = q.shape
        if q.dtype == torch.float32 and sdpa_type in ('memo_effi','flash'):
            sdpa_type = 'torch'
        if sdpa_type == 'math':
            attn = (q * scale) @ k.transpose(-2, -1)
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn = attn.masked_fill(~attn_mask, -torch.finfo(attn.dtype).max)
                else:
                    attn += attn_mask
            attn = attn.softmax(dim=-1)
            attn = attn_drop(attn)
            x = attn @ v
            return x.transpose(1, 2)
        elif sdpa_type == 'memo_effi':
            # if attn_mask is not None:
            #     L, S = q.size(-2), k.size(-2)
            #     if S % 8 != 0:
            #         # 创建一个更大的张量并进行切片以确保内存对齐
            #         attn_bias = torch.zeros(B, 1, L, (S // 8 + 1) * 8, dtype=q.dtype, device=q.device)[:, :, :, :S]
            #     else:
            #         attn_bias = torch.zeros(B, 1, L, S, dtype=q.dtype, device=q.device)
            #     if attn_mask.dtype == torch.bool:
            #         attn_bias.masked_fill_(~attn_mask, -torch.finfo(q.dtype).max)
            #         attn_bias = repeat(attn_bias,'b 1 n l -> b h n l',h=H)
            #     else:
            #         _attn_bias = repeat(attn_bias,'b 1 n l -> b h n l',h=H)
            #         _attn_bias = _attn_bias + attn_mask
            #         if S % 8 != 0:
            #             # 创建一个更大的张量并进行切片以确保内存对齐
            #             attn_bias = torch.zeros(B, H, L, (S // 8 + 1) * 8, dtype=q.dtype, device=q.device)
            #             attn_bias[:, :, :, :S] = _attn_bias
            #             attn_bias = attn_bias[:, :, :, :S]
            # else:
            #     attn_bias = None
            # x = xops.memory_efficient_attention(
            #         q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
            #         attn_bias=attn_bias,
            #         p=attn_drop.p if training else 0.,
            #     )
            # return x
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p = attn_drop.p if training else 0.,
                )
            return x.transpose(1, 2)
        elif sdpa_type == 'flash':
            assert attn_mask is None
            # flash-attn
            x = flash_attn_func(
                q.transpose(1,2),
                k.transpose(1,2),
                v.transpose(1,2),
                dropout_p = attn_drop.p if training else 0.,
                deterministic=deterministic if training else False
            )
            return x
        elif sdpa_type == 'torch':
            #if attn_mask is not None or not sdpa_type == 'flash':
            #with sdpa_kernel(SDPBackend.MATH):
            x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p = attn_drop.p if training else 0.,
            )
            return x.transpose(1, 2)
        elif sdpa_type == 'torch_math':
            with sdpa_kernel(SDPBackend.MATH):
                x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p = attn_drop.p if training else 0.,
                )
                return x.transpose(1, 2)
        else:
            raise NotImplementedError

class CAttention(nn.Module):
    def __init__(
            self, 
            dim: int, 
            num_heads: int = 8, 
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            deterministic: bool = True, 
            sdpa_type: str = 'torch',
    )-> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.deterministic = deterministic

        self.fused_attn_env = suppress

        self.kv = nn.Linear(dim, dim * 2, bias = False)
        self.q = nn.Linear(dim, dim, bias = False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.sdpa_type = sdpa_type

    def forward(self,x: torch.Tensor, _q: torch.Tensor, attn_mask=None, return_attn=False,freqs_cos=None,freqs_sin=None):
        B, N, C = x.shape
        # flash-attn
        if return_attn or self.sdpa_type == 'math':
            q = self.q(_q).reshape(B, 1, self.num_heads, self.head_dim).transpose(1,2)
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            if freqs_cos is not None:
                raise NotImplementedError
                q[:,1:] = q[:,1:] * freqs_cos + rotate_half(q[:,1:]) * freqs_sin
                k[:,1:] = k[:,1:] * freqs_cos + rotate_half(k[:,1:]) * freqs_sin

            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn.masked_fill(~attn_mask, -torch.finfo(attn.dtype).max)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2).reshape(B, 1, C)
        else:
            # torch
            if attn_mask is not None or not self.sdpa_type == 'flash':
                q = self.q(_q).reshape(B, 1, self.num_heads, self.head_dim).transpose(1,2)
                kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                k, v = kv.unbind(0)
                
                if freqs_cos is not None:
                    raise NotImplementedError
                    q[:,1:] = q[:,1:] * freqs_cos + rotate_half(q[:,1:]) * freqs_sin
                    k[:,1:] = k[:,1:] * freqs_cos + rotate_half(k[:,1:]) * freqs_sin

                with self.fused_attn_env():
                    x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
                x = x.transpose(1, 2).reshape(B, 1, C)

            elif self.sdpa_type == 'flash':
                # flash-attn
                q = self.q(_q).reshape(B, 1, self.num_heads, self.head_dim)
                kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
                k, v = kv.unbind(0)
                if freqs_cos is not None:
                    raise NotImplementedError
                    q[:,1:] = q[:,1:] * freqs_cos + rotate_half(q[:,1:]) * freqs_sin
                    k[:,1:] = k[:,1:] * freqs_cos + rotate_half(k[:,1:]) * freqs_sin
                x = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                    deterministic=self.deterministic if self.training else False
                )
                x = x.reshape(B,1,C)
            else:
                raise NotImplementedError

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            # cls token attn
            return x,attn[:,:,0,:],v

        return x
class Attention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            deterministic: bool = True,
            sdpa_type: str = 'torch',
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.deterministic = deterministic
        # ~~torch 2.5.1目前只有MATH才有复现性~~
        # 在设定torch.use_deterministic_algorithms(True) 后，torch自己会选择判定性算法，并且MEMO_EFFI（设定后）是可复现的
        #self.fused_attn_env = partial(sdpa_kernel,SDPBackend.MATH) if self.deterministic else suppress
        # MATH实现变动太快，2.4和2.5都不一样，数值不一样，显存消耗也不一样，不如直接用代码实现
        #self.fused_attn_env = partial(sdpa_kernel,SDPBackend.MATH) if sdpa_type == 'math' else suppress
        self.fused_attn_env = suppress
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sdpa_type = sdpa_type

    def forward(self, x: torch.Tensor, attn_mask=None, return_attn=False,freqs_cos=None,freqs_sin=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if freqs_cos is not None:
            q_nocls,k_nocls = q[:,:,1:],k[:,:,1:]
            freqs_cos,freqs_sin = freqs_cos.type(q_nocls.dtype),freqs_sin.type(q_nocls.dtype)
            q_nocls = q_nocls * freqs_cos + rotate_half(q_nocls) * freqs_sin
            k_nocls = k_nocls * freqs_cos + rotate_half(k_nocls) * freqs_sin
            
            q = torch.cat([q[:,:,0].unsqueeze(-2),q_nocls], dim=2)
            k = torch.cat([k[:,:,0].unsqueeze(-2),k_nocls], dim=2)

        # flash-attn
        if return_attn:
            # B H N D
            q = q * self.scale
            # Only use CLS token query (first token)
            cls_q = q[:, :, 0:1, :]  # B H 1 D
            
            # Compute attention only for CLS token
            attn = cls_q @ k.transpose(-2, -1)  # B H 1 N
            
            if attn_mask is not None:
                # Apply mask only to CLS attention
                attn = attn.masked_fill(~attn_mask[:, :, 0:1, :], -torch.finfo(attn.dtype).max)
            
            attn = attn.softmax(dim=-1)  # B H 1 N
            attn = self.attn_drop(attn)
            # 这里为了节省显存，只返回cls token的attn，不对x做MSA
            # attn = q @ k.transpose(-2, -1)
            # if attn_mask is not None:
            #     attn = attn.masked_fill(~attn_mask, -torch.finfo(attn.dtype).max)
            # attn = attn.softmax(dim=-1)
            # attn = self.attn_drop(attn)
            # x = attn @ v
            # x = x.transpose(1, 2).reshape(B, N, C)
        else:
            # torch
            #with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            if self.sdpa_type == 'math':
            #if return_attn:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                if attn_mask is not None:
                    attn = attn.masked_fill(~attn_mask, -torch.finfo(attn.dtype).max)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v
                x = x.transpose(1, 2).reshape(B, N, C)
            elif attn_mask is not None or not self.sdpa_type == 'flash':
                # if self.sdpa_type == 'torch':
                # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                # q, k, v = qkv.unbind(0)
                # q, k = self.q_norm(q), self.k_norm(k)
                # if freqs_cos is not None:
                #     q[:,1:] = q[:,1:] * freqs_cos + rotate_half(q[:,1:]) * freqs_sin
                #     k[:,1:] = k[:,1:] * freqs_cos + rotate_half(k[:,1:]) * freqs_sin
                with self.fused_attn_env():
                    x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
                x = x.transpose(1, 2).reshape(B, N, C)

                # 和torch实现一模一样
                # elif self.sdpa_type == 'effi_memo':
                #     qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
                #     q, k, v = qkv.unbind(0)
                #     if attn_mask is not None:
                #         L, S = q.size(1), k.size(1)
                #         if S % 8 != 0:
                #             # 创建一个更大的张量并进行切片以确保内存对齐
                #             attn_bias = torch.zeros(B, 1, L, (S // 8 + 1) * 8, dtype=q.dtype, device=q.device)[:, :, :, :S]
                #         else:
                #             attn_bias = torch.zeros(B, 1, L, S, dtype=q.dtype, device=q.device)
                #         attn_bias.masked_fill_(~attn_mask, -torch.finfo(q.dtype).max)
                #         attn_bias = repeat(attn_bias,'b 1 n l -> b h n l',h=self.num_heads)
                #     else:
                #         attn_bias = None
                #     x = xops.memory_efficient_attention(
                #             q, k, v,
                #             attn_bias=attn_bias,
                #             p=self.attn_drop.p if self.training else 0.,
                #         )
                #     x = x.reshape(B, N, C)

            elif self.sdpa_type == 'flash':
                # flash-attn
                # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                # q, k, v = qkv.unbind(0)
                # q, k = self.q_norm(q), self.k_norm(k)

                # if freqs_cos is not None:
                #     q[:,:,1:] = q[:,:,1:] * freqs_cos + rotate_half(q[:,:,1:]) * freqs_sin
                #     k[:,:,1:] = k[:,:,1:] * freqs_cos + rotate_half(k[:,:,1:]) * freqs_sin

                x = flash_attn_func(
                    q.transpose(1,2),
                    k.transpose(1,2),
                    v.transpose(1,2),
                    dropout_p=self.attn_drop.p if self.training else 0.,
                    deterministic=self.deterministic if self.training else False
                )
                x = x.reshape(B,N,C)
            else:
                raise NotImplementedError

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            # cls token attn
            return x,attn[:,:,0,1:],v

        return x
    
class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,deterministic=True,dropout=0.,ffn=False,sdpa_type='torch',n_heads=8,no_norm=False,attn_type='sa',pos=''):
        super().__init__()
        self.norm = norm_layer(dim) if not no_norm else nn.Identity()
        self.n_heads = n_heads
        self.pos = pos
        self.rope = None
        if 'rope' in pos:
            self.rope = VisionRotaryEmbedding(dim // n_heads,online_rope=True,decouple=True,custom_freqs='ntk-aware')
        elif 'alibi' in pos:
            if 'learn' in pos:
                self.m = nn.Parameter(torch.ones(n_heads))
            else:
                self.register_buffer("m", get_alibi_slope(n_heads))
        if attn_type == 'sa':
            self.attn = Attention(
                dim = dim,
                num_heads = n_heads,
                attn_drop=dropout,
                deterministic=deterministic,
                sdpa_type = sdpa_type
            )
        elif attn_type == 'ca':
            self.attn = CAttention(
                dim = dim,
                num_heads = n_heads,
                attn_drop=dropout,
                deterministic=deterministic,
                sdpa_type = sdpa_type
            )
        else:
            raise NotImplementedError
        if ffn:
            self.norm2 = norm_layer(dim) if not no_norm else nn.Identity()
            self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * 4),
            act_layer=nn.GELU,
            )
        self.ffn = ffn

    def forward_ca(self,x,q=None,attn_mask=None,need_attn=False, need_v=False,pos=None):
        assert len(x.shape) == 3
       
        alibi_bias = None
        freqs_cos = freqs_sin = None

        if self.rope is not None:
            freqs_cos, freqs_sin = self.rope.online_get_2d_rope_from_grid(pos[:,1:].transpose(-1,-2), pos[:,0][..., [1, 0]].unsqueeze(0))
            freqs_cos, freqs_sin = freqs_cos.to(x.dtype), freqs_sin.to(x.dtype)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        elif 'alibi' in self.pos:
            alibi_bias = get_alibi_bias(m=None,pos=pos,cls_token=True,n_heads=self.n_heads,device=x.device)

        if q is None:
            q = x[:,0,:]
            x = x[:,1:,:]

        if need_attn:
            z,attn,v = self.attn(self.norm(x),self.norm(q),return_attn=need_attn,attn_mask=alibi_bias,freqs_cos=freqs_cos,freqs_sin=freqs_sin)
            q = q+z

            if self.ffn:
                q = q + self.mlp(self.norm2(q))

            if len(q.shape) == 2:
                q = q.unsequeeze(1)

            if need_v:
                return q,attn,v
            else:
                return q,attn
        else:
            q = q + self.attn(self.norm(x),self.norm(q),attn_mask=alibi_bias,freqs_cos=freqs_cos,freqs_sin=freqs_sin)
            if self.ffn:
                q = q + self.mlp(self.norm2(q))
            
            if len(q.shape) == 2:
                q = q.unsequeeze(1)

            return q

    def forward_sa(self,x, attn_mask=None,need_attn=False, need_v=False,pos=None):
        alibi_bias = None
        freqs_cos = freqs_sin = None

        if self.rope is not None:
            freqs_cos, freqs_sin = self.rope.online_get_2d_rope_from_grid(pos[:,1:].transpose(-1,-2), pos[:,0][..., [1, 0]].unsqueeze(0))
            freqs_cos, freqs_sin = freqs_cos.to(x.dtype), freqs_sin.to(x.dtype)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        elif 'alibi' in self.pos:
            alibi_bias = get_alibi_bias(m=self.m,pos=pos,cls_token=True,n_heads=self.n_heads,device=x.device)

        if need_attn:
            z,attn,v = self.attn(self.norm(x),return_attn=need_attn,attn_mask=alibi_bias,freqs_cos=freqs_cos,freqs_sin=freqs_sin)
            x = x+z

            if self.ffn:
                x = x + self.mlp(self.norm2(x))

            if need_v:
                return x,attn,v
            else:
                return x,attn
        else:
            x = x + self.attn(self.norm(x),attn_mask=alibi_bias,freqs_cos=freqs_cos,freqs_sin=freqs_sin)
            if self.ffn:
                x = x + self.mlp(self.norm2(x))
            return x  

    def forward(self, x, q=None, attn_mask=None,need_attn=False, need_v=False,pos=None):
        #x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        if isinstance(self.attn,CAttention):
            return self.forward_ca(x,q,attn_mask,need_attn,need_v,pos=pos)
        elif isinstance(self.attn,Attention):
            return self.forward_sa(x,attn_mask,need_attn,need_v,pos=pos)

class ViTMIL(nn.Module):
    def __init__(self, input_dim,n_classes,dropout,act,mil_norm=None,mil_bias=False,mil_cls_bias=True,inner_dim=512,pos=None,n_layers=1,deterministic=True,attn_dropout=0.,embed_feat=True,feat_embed_type='norm',embed_feat_mlp_ratio=4,ffn=False,sdpa_type='torch',n_heads=8,vit_norm=True,embed_norm_pos=0,attn_type='sa',**kwargs):
        super(ViTMIL, self).__init__()

        if mil_bias:
            mil_cls_bias = True

        if pos in ('ppeg','peg'):
            assert n_layers > 1
        
        if attn_type == 'ca':
            assert n_layers == 1
        
        self.embed_norm_pos = embed_norm_pos
        assert self.embed_norm_pos in (0,1)
        if mil_norm == 'ln':
            assert embed_norm_pos == 0

        self.pos = pos

        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=inner_dim)
        elif pos == 'sincos':
            self.pos_embedding = SINCOS()
        elif pos == 'peg':
            self.pos_embedding = PEG(512)
        else:
            self.pos_embedding = nn.Identity()

        # self.feature = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),nn.Dropout(0.25))
        self.feature = []
        self.mil_norm = mil_norm
        if mil_norm == 'bn':   
            self.norm1 = nn.BatchNorm1d(input_dim) if embed_norm_pos == 0 else nn.BatchNorm1d(inner_dim)
        elif mil_norm == 'ln':
            self.feature += [nn.LayerNorm(input_dim,bias=mil_bias)]
            self.norm1 = nn.Identity()
        else:
            self.norm1 = nn.Identity()

        if embed_feat:
            if feat_embed_type == 'norm':
                self.feature += [nn.Linear(input_dim, inner_dim,bias=mil_bias)]
                if act.lower() == 'gelu':
                    self.feature += [nn.GELU()]
                elif act.lower() == 'relu':
                    self.feature += [nn.ReLU()]
                if dropout:
                    self.feature += [nn.Dropout(dropout)]
            elif feat_embed_type == 'act_first':
                if act.lower() == 'gelu':
                    self.feature += [nn.GELU()]
                elif act.lower() == 'relu':
                    self.feature += [nn.ReLU()]
                self.feature += [nn.Linear(input_dim, inner_dim,bias=mil_bias)]
                if dropout:
                    self.feature += [nn.Dropout(dropout)]
            elif feat_embed_type == 'mlp':
                self.feature += [nn.Linear(input_dim, inner_dim // embed_feat_mlp_ratio,bias=mil_bias)]
                if act.lower() == 'gelu':
                    self.feature += [nn.GELU()]
                elif act.lower() == 'relu':
                    self.feature += [nn.ReLU()]
                if dropout:
                    self.feature += [nn.Dropout(dropout)]
                self.feature += [nn.Linear(inner_dim // embed_feat_mlp_ratio, inner_dim,bias=mil_bias)]
        
        self.feature = nn.Sequential(*self.feature) if len(self.feature) > 0 else nn.Identity()
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, inner_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=inner_dim,deterministic=deterministic,dropout=attn_dropout,ffn=ffn,sdpa_type=sdpa_type,n_heads=n_heads,no_norm=not vit_norm,attn_type=attn_type,pos=pos)
        if n_layers >= 2:
            self.layer2 = TransLayer(dim=inner_dim,deterministic=deterministic,dropout=attn_dropout,ffn=ffn,sdpa_type=sdpa_type,n_heads=n_heads,no_norm=not vit_norm,pos=pos)
        else:
            self.layer2 = nn.Identity()
        self.n_layers = n_layers
        self.norm = nn.LayerNorm(inner_dim,bias=mil_cls_bias) if vit_norm else nn.Identity()
        self.classifier = nn.Linear(inner_dim, self.n_classes,bias=mil_cls_bias)

        self.apply(initialize_weights)

    def forward(self, x,attn_mask=None,return_attn=False,return_act=False,return_feat=False,pos=None,return_img_feat=False,only_return_attn=False):
        
        if len(x.size()) == 2:
            x = x.unsqueeze(0)

        if pos is not None:
            if len(pos.shape) == 2:
                pos = pos.unsqueeze(0)

        batch, num_patches, C = x.shape 

        attn = []
        if self.embed_norm_pos == 0:
            if self.mil_norm == 'bn':
                x = torch.transpose(x, -1, -2)
                x = self.norm1(x)
                x = torch.transpose(x, -1, -2)

        x = self.feature(x) #[B, n, 512]
        if self.pos == 'sincos':
            x = self.pos_embedding(x,pos=pos)

        if self.embed_norm_pos == 1:
            if self.mil_norm == 'bn':
                x = torch.transpose(x, -1, -2)
                x = self.norm1(x)
                x = torch.transpose(x, -1, -2)

        # cls_token
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = batch)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # translayer1
        if return_attn:
            x,_attn,v = self.layer1(x,need_attn=True,need_v=True,pos=pos)
            attn.append(_attn.clone())
        else:
            x = self.layer1(x,pos=pos)

        # add pos embedding
        if x.size(1) > 1 and self.pos == 'ppeg':
            x[:,1:,:] = self.pos_embedding(x[:,1:,:])
        
        # translayer2
        if return_attn and self.n_layers == 2:
            #x,_attn,_ = self.layer2(x,need_attn=True,need_v=True)
            #attn.append(_attn.clone())
            pass
        elif self.n_layers == 2:
            x = self.layer2(x,pos=pos)

        #x = self.norm(x)

        if return_feat:
            assert x.size(1) > 1
            logits = self.classifier(self.norm(x[:,0,:]))
            logits = [logits,x[:,1:,:]]
        elif return_img_feat:
            x = self.norm(x[:,0,:])
            logits = self.classifier(x)
            logits = [logits,x]
        else:
            #---->cls_token
            logits = self.classifier(self.norm(x[:,0,:]))
        if return_attn:
            output = []
            #output.append(logits)
            # TODO:这边暂时只支持return_attn就只要attn，而不是两者都要，因为一些显存的考虑
            output.append(None)
            output.append(attn)
            if return_act:
                output.append(v)
            return output
        else:
            return logits
