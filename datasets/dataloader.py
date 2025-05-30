from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset_feat import FeatClsDataset,FeatSurvDataset
from .dataset_img import SurvLMDBDataset,ClsLMDBDataset,MultiScaleDataset
from .data_utils import *

def build_dataloader(args,dataset):
    if args.image_input:
        return build_img_dataloader(args,dataset)
    else:
        return build_feat_dataloader(args,dataset,prefetch=args.prefetch)

def build_img_dataloader(args,dataset):
    loader_kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory}
    if args.lmdb:
        loader_kwargs['collate_fn'] = collate_fn_nbs
        if args.num_workers > 1:
            loader_kwargs['prefetch_factor'] = args.prefetch_factor
       
    loader_kwargs_val = deepcopy(loader_kwargs)
    loader_kwargs_test = deepcopy(loader_kwargs)

    if args.num_workers_test is not None:
        loader_kwargs_test['num_workers'] = args.num_workers_test
        loader_kwargs_val['num_workers'] = args.num_workers_test
        if args.num_workers_test < 2:
            loader_kwargs_test['prefetch_factor'] = None
            loader_kwargs_val['prefetch_factor'] = None

    if args.image_input and args.batch_size > 1:
        if args.sel_type == 'random':
            loader_kwargs['collate_fn'] = collate_fn_img_batch
        else:
            loader_kwargs['collate_fn'] = collate_fn_img_batch_list

    keep_same_psize = args.same_psize
    if 'feat' in args.sel_type:
        loader_kwargs['worker_init_fn'] = update_worker_dict
        keep_same_psize = 0

    if args.datasets.lower().startswith('surv'):
        if args.lmdb:
            if args.mul_scale_input:
                keep_same_psize_sub = keep_same_psize // args.mul_scale_ratio
                keep_same_psize = keep_same_psize - keep_same_psize_sub
                _train_set_1 = SurvLMDBDataset(args,args.dataset_sub_root,dataset['train'],persistence=args.persistence,keep_same_psize=keep_same_psize_sub,mode="train",channels_last=args.channels_last,img_size=args.img_size,
                h5_root=args.h5_sub_path)
                _train_set_2 = SurvLMDBDataset(args,args.env_train,dataset['train'],persistence=args.persistence,keep_same_psize=keep_same_psize,mode="train",channels_last=args.channels_last,img_size=args.img_size,
                h5_root=args.h5_path)
                train_set = MultiScaleDataset(_train_set_1,_train_set_2)
            else:
                train_set = SurvLMDBDataset(args,args.env_train,dataset['train'],persistence=args.persistence,keep_same_psize=keep_same_psize,mode="train",channels_last=args.channels_last,img_size=args.img_size,
                h5_root=args.h5_path)

            _test_img_size = 224
            test_set = SurvLMDBDataset(args,args.env,dataset['test'],persistence=args.persistence,channels_last=args.channels_last,mode="test",h5_root=args.h5_path,img_size=_test_img_size)
            if args.val_ratio != 0.:
                val_set = SurvLMDBDataset(args,args.env,dataset['val'],persistence=args.persistence,channels_last=args.channels_last,mode="val",h5_root=args.h5_path,img_size=_test_img_size)
            else:
                val_set = SurvLMDBDataset(args,args.env,dataset['test'],persistence=args.persistence,channels_last=args.channels_last,mode="val",h5_root=args.h5_path,img_size=_test_img_size)
    else:       
        if args.lmdb:
            if args.mul_scale_input:
                keep_same_psize_sub = keep_same_psize // args.mul_scale_ratio
                keep_same_psize = keep_same_psize - keep_same_psize_sub
                _train_set_1 = ClsLMDBDataset(args,args.dataset_sub_root,dataset['train'],persistence=args.persistence,keep_same_psize=keep_same_psize_sub,mode="train",_type=args.datasets,channels_last=args.channels_last,img_size=args.img_size,
                h5_root=args.h5_sub_path)
                _train_set_2 = ClsLMDBDataset(args,args.env_train,dataset['train'],persistence=args.persistence,keep_same_psize=keep_same_psize,mode="train",_type=args.datasets,channels_last=args.channels_last,img_size=args.img_size,
                h5_root=args.h5_path)
                train_set = MultiScaleDataset(_train_set_1,_train_set_2)
            else:
                train_set = ClsLMDBDataset(args,args.env_train,dataset['train'],persistence=args.persistence,keep_same_psize=keep_same_psize,mode="train",_type=args.datasets,channels_last=args.channels_last,img_size=args.img_size,
                h5_root=args.h5_path)

            _test_img_size = 224
            test_set = ClsLMDBDataset(args,args.env,dataset['test'],persistence=args.persistence,_type=args.datasets,channels_last=args.channels_last,mode="test",h5_root=args.h5_path,img_size=_test_img_size)
            if args.val_ratio != 0.:
                val_set = ClsLMDBDataset(args,args.env,dataset['val'],persistence=args.persistence,_type=args.datasets,channels_last=args.channels_last,mode="val",h5_root=args.h5_path,img_size=_test_img_size)
            else:
                val_set = ClsLMDBDataset(args,args.env,dataset['test'],persistence=args.persistence,_type=args.datasets,channels_last=args.channels_last,mode="val",h5_root=args.h5_path,img_size=_test_img_size)
                
        else:
            raise  NotImplementedError

    if args.distributed:
        train_loader = DataLoader(train_set, batch_size=args.batch_size,drop_last=args.drop_last,**loader_kwargs,sampler=DistributedSampler(train_set,shuffle=True))
        val_loader = DataLoader(val_set, batch_size=1,  **loader_kwargs_val,sampler=SequentialDistributedSampler(val_set,batch_size=1))
        test_loader = DataLoader(test_set, batch_size=1,  **loader_kwargs_test,sampler=SequentialDistributedSampler(test_set,batch_size=1))
    else: 
        train_loader = DataLoader(train_set, batch_size=args.batch_size,shuffle=True, drop_last=args.drop_last,**loader_kwargs)
        val_loader = DataLoader(val_set, batch_size=1,shuffle=False, **loader_kwargs_val)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **loader_kwargs_test)

    if args.prefetch:
        # Setting fp16 here will affect performance
        train_loader = PrefetchLoader(train_loader,device=args.device,need_transform=args.img_transform != 'none', transform_type=args.img_transform,img_size=args.img_size,trans_chunk=args.img_trans_chunk,crop_scale=args.crop_scale,load_gpu_later=args.load_gpu_later_train)
        assert not args.no_prefetch_test
        val_loader = PrefetchLoader(val_loader,device=args.device,is_train=False,load_gpu_later=args.load_gpu_later,trans_chunk=args.img_trans_chunk)
        test_loader = PrefetchLoader(test_loader,device=args.device, is_train=False,load_gpu_later=args.load_gpu_later,trans_chunk=args.img_trans_chunk)
        
    return train_loader,val_loader,test_loader

def update_worker_dict(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.update_imgs()

def _get_feat_dataloader(args,dataset,root,train=True,prefetch=True,sub_root=None):

    keep_same_psize = args.same_psize
    if 'feat' in args.sel_type:
        # Current implementation does not support multi-scale feature selection
        assert not args.mul_scale_input
        keep_same_psize = 0

    if args.datasets.lower().startswith('surv'):
        if train:
            if args.mul_scale_input:
                keep_same_psize_sub = keep_same_psize // args.mul_scale_ratio
                keep_same_psize = keep_same_psize - keep_same_psize_sub
                _train_set_1 = FeatSurvDataset(dataset,root=sub_root,persistence=args.persistence,keep_same_psize=keep_same_psize_sub,is_train=True,args=args)
                _train_set_2 = FeatSurvDataset(dataset,root=root,persistence=args.persistence,keep_same_psize=keep_same_psize,is_train=True,args=args)
                _dataset = MultiScaleDataset(_train_set_1,_train_set_2,image_input=False)
            else:
                _dataset = FeatSurvDataset(dataset,root=root,persistence=args.persistence,keep_same_psize=keep_same_psize,is_train=True,args=args)
        else:
            _dataset = FeatSurvDataset(dataset,root=root,persistence=args.persistence,is_train=False)
    else:
        p,l = dataset
        if train:
            if args.mul_scale_input:
                keep_same_psize_sub = keep_same_psize // args.mul_scale_ratio
                keep_same_psize = keep_same_psize - keep_same_psize_sub
                _train_set_1 = FeatClsDataset(p,l,sub_root,persistence=args.persistence,keep_same_psize=keep_same_psize_sub,is_train=True,_type=args.datasets,args=args)
                _train_set_2 = FeatClsDataset(p,l,root,persistence=args.persistence,keep_same_psize=keep_same_psize,is_train=True,_type=args.datasets,args=args)
                _dataset = MultiScaleDataset(_train_set_1,_train_set_2,image_input=False)
            else:
                _dataset = FeatClsDataset(p,l,root,persistence=args.persistence,keep_same_psize=keep_same_psize,is_train=True,_type=args.datasets,args=args)
        else:
            _dataset = FeatClsDataset(p,l,root,persistence=args.persistence,_type=args.datasets,args=args)

    loader_kwargs = {'pin_memory':args.pin_memory}

    if train:
        _dataloader = DataLoader(_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last,num_workers=args.num_workers,**loader_kwargs)
    else:
        _num_workers_test = args.num_workers_test or args.num_workers
        _dataloader = DataLoader(_dataset, batch_size=1, shuffle=False, num_workers=_num_workers_test,pin_memory=args.pin_memory)

    if prefetch:
        _dataloader = PrefetchLoader(_dataloader,device=args.device,need_norm=False)

    return _dataloader

def build_feat_dataloader(args,dataset,mode='all',root=None,prefetch=True):
    root = root or args.dataset_root

    if not args.datasets.lower().startswith('surv'):
        train_p, train_l, test_p, test_l,val_p,val_l = parse_dataframe(args,dataset)
        dataset = {'train':[train_p, train_l],"val":[val_p,val_l],"test":[test_p, test_l]}

    if mode == 'all':
        train_loader = _get_feat_dataloader(args,dataset['train'],root=root,train=True,prefetch=prefetch,sub_root=args.dataset_sub_root if args.mul_scale_input else None)
        if args.val_ratio == 0.:
            val_loader = _get_feat_dataloader(args,dataset['test'],root=root,train=False,prefetch=prefetch)
        else:
            val_loader = _get_feat_dataloader(args,dataset['val'],root=root,train=False,prefetch=prefetch)
        test_loader = _get_feat_dataloader(args,dataset['test'],root=root,train=False,prefetch=prefetch)
    elif mode == 'test':
        test_loader = _get_feat_dataloader(args,dataset['test'],root=root,train=False,prefetch=prefetch)
        train_loader = val_loader = None
    elif mode == 'no_train':
        if args.val_ratio == 0.:
            val_loader = _get_feat_dataloader(args,dataset['test'],root=root,train=False,prefetch=prefetch)
        else:
            val_loader = _get_feat_dataloader(args,dataset['val'],root=root,train=False,prefetch=prefetch)
        test_loader = _get_feat_dataloader(args,dataset['test'],root=root,train=False,prefetch=prefetch)
        train_loader = None
    elif mode == 'train':
        train_loader = _get_feat_dataloader(args,dataset['train'],root=root,train=True,prefetch=prefetch,sub_root=args.dataset_sub_root if args.mul_scale_input else None)
        val_loader = test_loader = None
    else:
        raise NotImplementedError

    return train_loader,val_loader,test_loader

