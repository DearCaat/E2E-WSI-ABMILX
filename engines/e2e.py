import torch
from .common_mil import CommonMIL
from modules.e2e import Normalize
from contextlib import suppress
from timm.utils import unwrap_model

class E2E(CommonMIL):
    def __init__(self, args,device) -> None:
        super().__init__(args)
        self.dataset = None
        self.loader = None
        self.device = device
        self.max_psize = args.max_patch_train or args.same_psize
        self.norm = Normalize(device,args.channels_last)
        self.static_pos=[self.max_psize]
        self.inference_mode = torch.inference_mode if args.freeze_enc else suppress

    def init_func_train(self,args,model,others,epoch,optimizer,**kwargs):
        self.training = True

    def after_get_data_func(self,args,device,bag,optimizer,loader,n_iter,epoch,model,**kwargs):
        pass

    def get_grad_patch(self,args,bag,device,others,pos,feat):

        patch_num = bag.size(0)
        if patch_num > self.max_psize:
            if args.sel_type == 'random':
                indices = torch.randperm(patch_num, device=bag.device)
                selected_indices = indices[:self.max_psize]
                remaining_indices = indices[self.max_psize:]

            elif args.sel_type == 'ema':
                with torch.no_grad():
                    selected_indices,remaining_indices = others['model_ema'].module.select_patch(bag,self.max_psize)
                    selected_indices = selected_indices.to('cpu')
                    remaining_indices = remaining_indices.to('cpu')

            selected_patches = bag[selected_indices]
            if pos is not None:
                pos = torch.cat([pos[0].unsqueeze(0),pos[1:][selected_indices]],dim=0)

            if feat is not None:
                feat = feat[remaining_indices]

            remaining_patches = bag[remaining_indices]
            ori_indices = None
            if not args.prefetch:
                selected_patches = selected_patches.to(device, non_blocking=True)
                remaining_patches = remaining_patches.to(device, non_blocking=True)
                del bag
                selected_patches = self.norm(selected_patches)
                remaining_patches = self.norm(remaining_patches)
        elif patch_num < self.max_psize:
            if args.pad_enc_bs or (args.sel_type == 'ema' and args.batch_size > 1):
                ori_indices = torch.arange(patch_num, device=bag.device)
                selected_indices = torch.cat([ori_indices, torch.randint(0, patch_num, (self.max_psize - patch_num,), device=bag.device)])
                selected_patches = bag[selected_indices]
                if pos is not None:
                    pos = torch.cat([pos[0].unsqueeze(0),pos[1:][selected_indices]],dim=0)
            else:
                selected_patches = bag
                ori_indices = None
            remaining_patches = None
        else:
            # If the number of patches is less than or equal to max_psize, all patches are in selected_patches
            selected_patches = bag
            remaining_patches = None
            ori_indices = None
            if not args.prefetch:
                selected_patches = selected_patches.to(device, non_blocking=True)
                del bag
                selected_patches = self.norm(selected_patches)

        remaining_patches = None

        return selected_patches,remaining_patches,ori_indices,pos,feat

    def forward_func(self,args,model,model_ema,bag,label,criterion,batch_size,i,loader,device,others,pos=None,idx=None,feat=None,**kwargs):
        
        ps = None
        # Simple implementation first, consider optimization later
        if not args.sel_type == 'ema':
            if type(bag) in (tuple,list):
                # Change to pos input, temporarily not considering feat input
                # feat input
                bag,ps = bag
                
            if len(bag.size()) == 5:
                bag = bag.squeeze(0)

            _, C, W, H = bag.size()

        if args.batch_size > 1:
            if args.sel_type == 'ema':
                # Preallocate memory for batched selected patches
                B = len(bag)
                selected_patches = torch.empty((B*self.max_psize, 3, 224, 224), 
                                                device=device,
                                                memory_format=torch.channels_last)
                
                # Process each bag in the batch
                for b_idx, b in enumerate(bag):
                    sel_p, _, _, _, _ = self.get_grad_patch(args, b, device, others, None, None)
                    if len(sel_p.shape) == 5:
                        sel_p = sel_p.squeeze(0)
                    selected_patches[b_idx*self.max_psize:(b_idx+1)*self.max_psize].copy_(sel_p)

                patch_num = bag[0].size(0)
                keep_num = self.max_psize
                ori_indices = None
                remaining_patches = None
            else:
                patch_num = bag.size(0) / batch_size
                keep_num = patch_num
                ori_indices = None
                selected_patches = bag
                remaining_patches = None
        else:
            if ps is None:
                patch_num = bag.size(0)
                selected_patches,remaining_patches,ori_indices,pos,feat = self.get_grad_patch(args,bag,device,others,pos,feat)
                keep_num = selected_patches.size(0)
                if pos is not None:
                    if len(pos.shape) == 2:
                        pos = pos.unsqueeze(0)
            else:
                raise NotImplementedError

        if 'dsmil' in args.model or 'clam' in args.model:
            logits,aux_loss,_ = model((selected_patches,remaining_patches),ps=ori_indices,B=batch_size,label=label,loss=criterion,pos=pos)
        else:
            logits = model((selected_patches,remaining_patches),ori_indices,pos=pos,B=batch_size,feat=feat)
            aux_loss = 0.

        return logits,label,aux_loss,patch_num,keep_num,0.
    
    def after_backward_func(self,args,model,others,num_updates,**kwargs):
        if 'model_ema' in others:
            others['model_ema'].update(model, step=num_updates)
    
    def final_train_func(self,args,model):
        
        if args.no_ddp_broad_buf:
            for buf_name, buf_val in unwrap_model(model).named_buffers(recurse=True):
                torch.distributed.broadcast(buf_val, 0)

    def validate_func(self,args,model,bag,label,criterion,batch_size,i,loader,device,others,**kwargs):
        
        if type(bag) in (tuple,list):
            if isinstance(bag[1], torch.Tensor):
                bag, pos = bag
                ps = None
            # batch input
            else:
                bag,ps = bag
                pos = None
        else:
            ps = None
            pos = None

        if len(bag.size()) == 5:
            bag = bag.squeeze(0)

        patch_num = bag.size(0)
        all_patches = bag
        del bag

        keep_num = all_patches.size(0)

        if args.test_type != 'main':
            logits_ema = others['model_ema'](all_patches,pos=pos)
            logits_ema = logits_ema[0] if 'dsmil' in args.model else logits_ema
            if args.test_type == 'ema':
                logits = logits_ema
            elif args.test_type == 'both':
                logits = model(all_patches,pos=pos)
                logits = logits[0] if 'dsmil' in args.model else logits
                logits = [logits,logits_ema]
            elif args.test_type == 'both_ema':
                logits = model(all_patches,pos=pos)
                logits = logits[0] if 'dsmil' in args.model else logits
                logits = [logits_ema,logits]
        else:
            logits = model(all_patches,pos=pos)
            logits = logits[0] if 'dsmil' in args.model else logits
        
        return logits,label
    
    def init_func_val(self,args,amp_autocast,model,loader,status='val',others=None,epoch=0):
        pass