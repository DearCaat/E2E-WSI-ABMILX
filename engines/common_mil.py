class CommonMIL():
	def __init__(self,args) -> None:
		self.training = True

	def init_func_train(self,args,**kwargs):
		self.training = True
	
	def init_func_val(self,args,**kwargs):
		self.training = False
	
	def after_get_data_func(self,args,**kwargs):
		pass

	def forward_func(self,args,model,model_ema,bag,label,criterion,batch_size,i,epoch,n_iter,pos,**kwargs):
		pad_ratio=0.
		if args.model in ('clam_sb','clam_mb','dsmil'):
			logits, aux_loss, _ = model(bag,label=label,loss=criterion,pos=pos)
			keep_num = patch_num = bag.size(1)
		else:
			logits = model(bag,pos=pos)
			aux_loss, patch_num, keep_num = 0., bag.size(1), bag.size(1)

		return logits,label,aux_loss,patch_num,keep_num,pad_ratio
	
	def after_backward_func(self,args,**kwargs):
		pass
	
	def final_train_func(self,args,**kwargs):
		pass

	def validate_func(self,args,model,bag,label,criterion,batch_size,i,pos,**kwargs):
		if args.model == 'dsmil':
			logits,_ = model(bag,pos=pos)
		else:
			logits = model(bag,pos=pos)
		
		return logits,label