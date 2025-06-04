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
        self.max_psize = args.max_patch_train or args.same_psize # Maximum patch size for training
        self.norm = Normalize(device,args.channels_last) # Normalization module for patches
        self.static_pos=[self.max_psize] # Static position, seems related to patch selection or positional encoding
        self.inference_mode = torch.inference_mode if args.freeze_enc else suppress # Context manager for inference mode

    def init_func_train(self,args,model,others,epoch,optimizer,**kwargs):
        self.training = True # Set training mode

    def after_get_data_func(self,args,device,bag,optimizer,loader,n_iter,epoch,model,**kwargs):
        pass # Placeholder for actions after data is fetched

    def get_grad_patch(self,args,bag,device,others,pos,feat):
        """Selects patches for gradient computation based on selection type."""
        patch_num = bag.size(0) # Total number of patches in the bag
        if patch_num > self.max_psize:
            # If number of patches exceeds max_psize, select a subset
            if args.sel_type == 'random':
                # Randomly select patches
                indices = torch.randperm(patch_num, device=bag.device)
                selected_indices = indices[:self.max_psize]
                remaining_indices = indices[self.max_psize:]

            elif args.sel_type == 'ema':
                # Select patches using EMA model (presumably for importance sampling)
                with torch.no_grad():
                    selected_indices,remaining_indices = others['model_ema'].module.select_patch(bag,self.max_psize)
                    selected_indices = selected_indices.to('cpu') # Move indices to CPU
                    remaining_indices = remaining_indices.to('cpu')

            selected_patches = bag[selected_indices] # Get selected patches
            if pos is not None:
                # Update positional information for selected patches
                pos = torch.cat([pos[0].unsqueeze(0),pos[1:][selected_indices]],dim=0)

            if feat is not None:
                # Update feature information for remaining patches (if any)
                feat = feat[remaining_indices]

            remaining_patches = bag[remaining_indices] # Get remaining patches
            ori_indices = None # Original indices not tracked here for this branch
            if not args.prefetch:
                # If not prefetching, move patches to device and normalize
                selected_patches = selected_patches.to(device, non_blocking=True)
                remaining_patches = remaining_patches.to(device, non_blocking=True)
                del bag # Free memory
                selected_patches = self.norm(selected_patches)
                remaining_patches = self.norm(remaining_patches)
        elif patch_num < self.max_psize:
            # If number of patches is less than max_psize, pad if necessary
            if args.pad_enc_bs or (args.sel_type == 'ema' and args.batch_size > 1):
                # Pad by duplicating existing patches randomly
                ori_indices = torch.arange(patch_num, device=bag.device)
                selected_indices = torch.cat([ori_indices, torch.randint(0, patch_num, (self.max_psize - patch_num,), device=bag.device)])
                selected_patches = bag[selected_indices]
                if pos is not None:
                    # Update positional information for padded patches
                    pos = torch.cat([pos[0].unsqueeze(0),pos[1:][selected_indices]],dim=0)
            else:
                selected_patches = bag # Use all patches
                ori_indices = None
            remaining_patches = None # No remaining patches in this case
        else:
            # If number of patches is equal to max_psize
            selected_patches = bag # Use all patches
            remaining_patches = None
            ori_indices = None
            if not args.prefetch:
                # If not prefetching, move patches to device and normalize
                selected_patches = selected_patches.to(device, non_blocking=True)
                del bag # Free memory
                selected_patches = self.norm(selected_patches)

        remaining_patches = None # Ensure remaining_patches is None if not used (or handled above)

        return selected_patches,remaining_patches,ori_indices,pos,feat

    def forward_func(self,args,model,model_ema,bag,label,criterion,batch_size,i,loader,device,others,pos=None,idx=None,feat=None,**kwargs):
        """Main forward pass logic for training."""
        ps = None # Placeholder for patch selection indices or similar
        # Simple implementation first, consider optimization later
        if not args.sel_type == 'ema': # If not using EMA for selection
            if type(bag) in (tuple,list):
                # If bag is a tuple/list, it might contain (patches, positions) or (patches, features)
                # Change to pos input, temporarily not considering feat input
                # feat input
                bag,ps = bag # Unpack bag and patch selection info (ps)
                
            if len(bag.size()) == 5: # If bag has 5 dimensions (e.g., [1, N, C, H, W]), squeeze the first dim
                bag = bag.squeeze(0)

            _, C, W, H = bag.size() # Get patch dimensions

        if args.batch_size > 1:
            if args.sel_type == 'ema':
                # Handle batch processing with EMA selection
                B = len(bag) # Batch size (number of bags)
                # Preallocate memory for batched selected patches
                selected_patches = torch.empty((B*self.max_psize, 3, 224, 224), 
                                                device=device,
                                                memory_format=torch.channels_last) # Use channels_last for potential performance benefits
                
                # Process each bag in the batch
                for b_idx, b_item in enumerate(bag):
                    # Get selected patches for the current bag item
                    sel_p, _, _, _, _ = self.get_grad_patch(args, b_item, device, others, None, None)
                    if len(sel_p.shape) == 5:
                        sel_p = sel_p.squeeze(0) # Squeeze if necessary
                    selected_patches[b_idx*self.max_psize:(b_idx+1)*self.max_psize].copy_(sel_p) # Copy selected patches to preallocated tensor

                patch_num = bag[0].size(0) # Number of patches in the first bag (assuming all bags have same original patch num before selection)
                keep_num = self.max_psize # Number of patches kept per bag
                ori_indices = None # Original indices not tracked for batched EMA
                remaining_patches = None
            else:
                # Handle batch processing without EMA selection (all patches are used or randomly selected if get_grad_patch is called)
                patch_num = bag.size(0) / batch_size # Average number of patches per bag in the batch
                keep_num = patch_num # All patches are kept (or selected by get_grad_patch if called)
                ori_indices = None
                selected_patches = bag # The input 'bag' is already the batch of selected patches
                remaining_patches = None
        else: # Single bag processing (batch_size == 1)
            if ps is None: # If patch selection info (ps) is not provided directly
                patch_num = bag.size(0) # Total number of patches in the bag
                # Select patches for gradient computation
                selected_patches,remaining_patches,ori_indices,pos,feat = self.get_grad_patch(args,bag,device,others,pos,feat)
                keep_num = selected_patches.size(0) # Number of patches kept
                if pos is not None:
                    if len(pos.shape) == 2: # Ensure pos has a batch dimension
                        pos = pos.unsqueeze(0)
            else:
                raise NotImplementedError # Handling for pre-selected patches (ps) not fully implemented here

        # Forward pass through the model
        if 'dsmil' in args.model or 'clam' in args.model:
            # For DS-MIL or CLAM models, which might have specific input formats or auxiliary losses
            logits,aux_loss,_ = model((selected_patches,remaining_patches),ps=ori_indices,B=batch_size,label=label,loss=criterion,pos=pos)
        else:
            # For other models
            logits = model((selected_patches,remaining_patches),ori_indices,pos=pos,B=batch_size,feat=feat)
            aux_loss = 0. # No auxiliary loss by default for other models

        return logits,label,aux_loss,patch_num,keep_num,0. # Return model outputs and related info
    
    def after_backward_func(self,args,model,others,num_updates,**kwargs):
        if 'model_ema' in others:
            others['model_ema'].update(model, step=num_updates) # Update EMA model after backward pass
    
    def final_train_func(self,args,model):
        
        if args.no_ddp_broad_buf:
            # Broadcast buffers in DistributedDataParallel (DDP) if specified
            for buf_name, buf_val in unwrap_model(model).named_buffers(recurse=True):
                torch.distributed.broadcast(buf_val, 0)

    def validate_func(self,args,model,bag,label,criterion,batch_size,i,loader,device,others,**kwargs):
        """Validation forward pass logic."""
        if type(bag) in (tuple,list):
            if isinstance(bag[1], torch.Tensor):
                # Input is (patches, positions)
                bag, pos = bag
                ps = None
            # batch input
            else:
                # Input is (patches, patch_selection_info) or similar
                bag,ps = bag
                pos = None
        else:
            ps = None # No patch selection info
            pos = None # No positional info

        if len(bag.size()) == 5: # If bag has 5 dimensions, squeeze the first one
            bag = bag.squeeze(0)

        patch_num = bag.size(0) # Total number of patches
        all_patches = bag # Use all patches for validation
        del bag # Free memory

        keep_num = all_patches.size(0) # Number of patches kept (all of them)

        # Determine which model to use for validation (main model, EMA model, or both)
        if args.test_type != 'main':
            logits_ema = others['model_ema'](all_patches,pos=pos) # Get logits from EMA model
            logits_ema = logits_ema[0] if 'dsmil' in args.model else logits_ema # Adjust output format for DS-MIL
            if args.test_type == 'ema':
                logits = logits_ema # Use only EMA model logits
            elif args.test_type == 'both':
                logits_main = model(all_patches,pos=pos) # Get logits from main model
                logits_main = logits_main[0] if 'dsmil' in args.model else logits_main
                logits = [logits_main,logits_ema] # Return logits from both models
            elif args.test_type == 'both_ema': # Similar to 'both', but EMA logits first
                logits_main = model(all_patches,pos=pos)
                logits_main = logits_main[0] if 'dsmil' in args.model else logits_main
                logits = [logits_ema,logits_main]
        else:
            # Use only main model for validation
            logits = model(all_patches,pos=pos)
            logits = logits[0] if 'dsmil' in args.model else logits # Adjust output format for DS-MIL
        
        return logits,label # Return logits and labels
    
    def init_func_val(self,args,amp_autocast,model,loader,status='val',others=None,epoch=0):
        pass