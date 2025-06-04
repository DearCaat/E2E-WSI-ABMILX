import torch
import torch.nn as nn
import numpy as np

from contextlib import suppress
from datasets.data_utils import IMAGENET_MEAN,IMAGENET_STD

class Normalize(nn.Module):
    def __init__(self,device,channels_last):
        super(Normalize,self).__init__()
        self.mean = torch.tensor(
            [x * 255 for x in IMAGENET_MEAN], device=device).view(1,3, 1, 1).to(memory_format=torch.channels_last)
        self.std = torch.tensor(
            [x * 255 for x in IMAGENET_STD], device=device).view(1,3, 1, 1).to(memory_format=torch.channels_last)
        
        if channels_last:
            self.mean.to(memory_format=torch.channels_last)
            self.std.to(memory_format=torch.channels_last)

    def __call__(self, tensor):
        return tensor.sub_(self.mean).div_(self.std)

def select_mask_fn(x, score, largest, mask_ratio=0., k=None, random_ratio=1., msa_fusion='vote'):
    """
    Selects indices based on scores, applying masking and random sampling.

    Args:
        x: Input tensor (batch, length, dim). Used for shape information.
        score: Attention scores used for selection.
        largest: Whether to select top-k largest (True) or smallest (False) scores.
        mask_ratio: Target ratio of elements to select.
        k: Number of top elements to select. If None, calculated based on mask_ratio.
        random_ratio: Ratio of selected elements to randomly sample.
        msa_fusion: Method for fusing MSA scores ('mean' or 'vote').

    Returns:
        selected_indices: Indices of selected elements.
    """
    N = x.shape[0]
    mask_ratio_ori = mask_ratio
    mask_ratio = mask_ratio / random_ratio
    if mask_ratio > 1:
        random_ratio = mask_ratio_ori
        mask_ratio = 1.
    random_k = None

    if k >= N:
        return list(range(N))

    if k is None:
        k = int(np.ceil((N * mask_ratio)))
    else:
        if int(k / random_ratio) >= N:
            random_k = k
            k = N
        else:
            k = int(k / random_ratio)

    if len(score.size()) > 2:
        if msa_fusion == 'mean':
            _, cls_attn_topk_idx = torch.topk(score, int(k // score.size(1)), largest=largest)
            cls_attn_topk_idx = torch.unique(cls_attn_topk_idx.flatten(-3, -1))
        elif msa_fusion == 'vote':
            vote = score.clone()
            vote[:] = 0
            _, idx = torch.topk(score, k=k, sorted=False, largest=largest)
            mask = torch.zeros_like(vote, dtype=torch.bool)  # Use boolean mask
            mask.scatter_(2, idx, True)
            vote[mask] = 1
            vote = vote.sum(dim=1)
            _, cls_attn_topk_idx = torch.topk(vote, k=k, sorted=False)
            cls_attn_topk_idx = cls_attn_topk_idx[0]  # Get indices from the first batch element
    else:
        _, cls_attn_topk_idx = torch.topk(score, k, largest=largest)
        cls_attn_topk_idx = cls_attn_topk_idx.squeeze(0)

    # Randomly sample
    if random_ratio < 1.:
        random_idx = torch.randperm(cls_attn_topk_idx.size(0), device=cls_attn_topk_idx.device)
        random_k = int(np.ceil((cls_attn_topk_idx.size(0) * random_ratio))) if random_k is None else random_k
        cls_attn_topk_idx = torch.gather(cls_attn_topk_idx, dim=0, index=random_idx[:random_k])

    return cls_attn_topk_idx  # Return the selected indices

class E2E(nn.Module):
    def __init__(self,encoder,mil,device,args=None,p_batch_size=2048,p_batch_size_v=2048,input_dim=1024,all_patch_train=False):
        super(E2E, self).__init__()
        self.encoder = encoder
        self.mil = mil

        if args is not None:
            self.p_batch_size = args.p_batch_size
            self.p_batch_size_v = args.p_batch_size_v
            self.n_classes = args.n_classes
            self.input_dim = args.input_dim
            self.inference_mode = suppress
            self.freeze_enc = False
            self.all_patch_train = args.all_patch_train
            self.load_gpu_later = args.load_gpu_later
            self.load_gpu_later_train = args.load_gpu_later_train
            self.device = args.device
        else:
            self.device = device
            self.p_batch_size = p_batch_size
            self.p_batch_size_v = p_batch_size_v
            self.input_dim = input_dim
            self.all_patch_train = all_patch_train

        if args.freeze_enc:
            self.freeze_encoder(True)
            self.inference_mode = torch.inference_mode
            if hasattr(self.encoder,'finetune'):
                self.encoder.finetune(False)

    def train(self,mode=True):
        if mode:
            self.training = True
            if self.freeze_enc:
                self.encoder.train(False)
                self.mil.train()
            else:
                for module in self.children():
                    module.train(mode)
        else:
            self.training = False
            for module in self.children():
                module.train(mode)

    def forward_encoder(self,x,**kwargs):
        if (self.load_gpu_later and not self.training) or self.load_gpu_later_train:
            x = x.to(self.device,non_blocking=True)
        return self.encoder(x,**kwargs)
    
    def freeze_encoder(self,mode=True):
        if mode is not None:
            self.freeze_enc = mode
        else:
            self.freeze_enc = not self.freeze_enc

        if self.freeze_enc:
            self.encoder.eval()
            self.inference_mode = torch.no_grad
            for name, parameter in self.encoder.named_parameters():
                parameter.requires_grad=False
        else:
            self.encoder.train()
            self.inference_mode = suppress
            for name, parameter in self.encoder.named_parameters():
                parameter.requires_grad=True

    @torch.compiler.disable()
    def forward_mil(self,x,return_img_feat=False,**kwargs):
        logits = self.mil(x,return_img_feat=return_img_feat,**kwargs)
        return logits
    
    @torch.compiler.disable(recursive=False)
    def forward_encoder_batch(self,x,batch_size,init_tensor_len=0,**enc_kwargs):
        enc_kwargs = {} if enc_kwargs is None else enc_kwargs
        keep_num = x.size(0)
        _device = self.device
        _feats = torch.empty((keep_num+init_tensor_len,self.input_dim),device=_device)
        _x = torch.split(x, batch_size)
        with self.inference_mode():
            for i,batch in enumerate(_x):
                if i == len(_x) - 1:
                    _feats[i*batch_size:keep_num] = self.forward_encoder(batch)
                else:
                    _feats[i*batch_size:(i+1)*batch_size] = self.forward_encoder(batch)
        return _feats

    @torch.no_grad()
    @torch.compiler.disable()
    def select_patch(self, x, k, encode=True, random_ratio=1.):
        ps = x.shape[0]
        _feats = self.forward_encoder_batch(x, self.p_batch_size_v) if encode else x.clone()
        _logits, score, act = self.mil(_feats, return_attn=True, return_act=True)
        
        if score is not None and isinstance(score,(list,tuple)):
            score = score[0]

        selected_indices = select_mask_fn(x, score, True, k=k, random_ratio=random_ratio)
        all_indices = torch.arange(ps, device=selected_indices.device)  # Create a tensor of all indices
        remaining_indices = all_indices[~torch.isin(all_indices, selected_indices)] # Efficiently find remaining indices

        return selected_indices, remaining_indices

    @torch.compiler.disable(recursive=False)
    def forward(self,x,ps=None,B=None,**mil_kwargs):
        x_grad = x
        enc_kwargs = {}

        if self.training:
            if len(x_grad) <= self.p_batch_size:
                _feats = self.forward_encoder(x_grad,**enc_kwargs)
            else:
                _feats = self.forward_encoder_batch(x_grad,self.p_batch_size,**enc_kwargs)
        else:
            _feats = self.forward_encoder_batch(x_grad,self.p_batch_size_v,**enc_kwargs)

        if ps is not None and not self.all_patch_train:
            _feats = _feats[ps]
        
        if B is not None:
            M=int(_feats.size(0)/B)
            return self.forward_mil(_feats.view(B,M,-1),**mil_kwargs)
        else:
            return self.forward_mil(_feats.unsqueeze(0),**mil_kwargs)
