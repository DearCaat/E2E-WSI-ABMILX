import argparse
import yaml
import torch
from timm.utils import init_distributed_device

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='Computional Pathology Training Script')

##### Dataset 
group = parser.add_argument_group('Dataset')
# Paths
parser.add_argument('--dataset_root', default='/data/xxx/TCGA', type=str, help='Dataset root path')
parser.add_argument('--dataset_sub_root', default=None, type=str, help='Dataset root path for multi-scale input')
group.add_argument('--csv_path', default=None, type=str, help='Dataset CSV path for Label and Split')
group.add_argument('--h5_path', default=None, type=str, help='Dataset H5 path for Coordinates')
group.add_argument('--h5_sub_path', default=None, type=str, help='Sub-dataset H5 path for Coordinates')
# Dataset settings
group.add_argument('--datasets', default='panda', type=str, help='Dataset')
group.add_argument('--val_ratio', default=0., type=float, help='Validation set ratio')
group.add_argument('--fold_start', default=0, type=int, help='Start validation fold [0]')
group.add_argument('--cv_fold', default=5, type=int, help='Number of cross validation folds [5]')
group.add_argument('--img_size', default=224, type=int, help='Image size [224]')
group.add_argument('--val2test', action='store_true', help='Use validation set as test set')
group.add_argument('--random_fold', action='store_true', help='Enable multi-fold random experiment')
group.add_argument('--random_seed', action='store_true', help='Enable random seed for multi-fold validation')
# Multi-scale input settings
group.add_argument('--mul_scale_input', action='store_true', help='Enable multi-scale input')
group.add_argument('--mul_scale_ratio', default=2., type=float, help='Multi-scale ratio [2.0]')
# Patch size settings
group.add_argument('--same_psize', default=0., type=float, help='Keep the same size for all patches [0]')
group.add_argument('--same_psize_pad_type', default='zero', type=str, choices=['zero','random','none'])
group.add_argument('--same_psize_ratio', default=0., type=float, help='Ratio for same patch size [0]')
# Input settings
group.add_argument('--image_input', action='store_true', help='Use image input instead of features')
# Dataloader settings
group.add_argument('--num_workers', default=6, type=int, help='Number of dataloader workers')
group.add_argument('--num_workers_test', default=None, type=int, help='Number of test dataloader workers')
group.add_argument('--lmdb', action='store_true', help='Enable LMDB Dataset')
group.add_argument('--pin_memory', action='store_true', help='Enable pinned memory')
group.add_argument('--no_prefetch', action='store_true', help='Disable prefetching')
group.add_argument('--no_prefetch_test', action='store_true', help='Disable prefetching for testing')
group.add_argument('--prefetch_factor', default=2, type=int, help='Prefetch factor [2]')
group.add_argument('--persistence', action='store_true', help='Enable persistent dataset caching')
# Image transformation settings
group.add_argument('--img_transform', default='none', type=str, help='Image transformation type')
group.add_argument('--img_trans_chunk', default=4, type=int, help='Image transformation chunk size')
group.add_argument('--crop_scale', default=0.08, type=float, help='Crop scale for transformation')

##### Training
group = parser.add_argument_group('Training')
group.add_argument('--main_alpha', default=1.0, type=float, help='Main loss weight')
group.add_argument('--aux_alpha', default=.0, type=float, help='Aux loss weight')
group.add_argument('--num_epoch', default=200, type=int, help='Total number of training epochs [200]')
group.add_argument('--epoch_start', default=0, type=int, help='Starting epoch number [0]')
group.add_argument('--early_stopping', action='store_false', help='Enable early stopping')
group.add_argument('--max_epoch', default=130, type=int, help='Maximum training epochs for early stopping [130]')
group.add_argument('--warmup_epochs', default=0, type=int, help='Number of warmup epochs [0]')
group.add_argument('--patient', default=20, type=int, help='Patience epochs for early stopping [20]')
group.add_argument('--input_dim', default=1024, type=int, help='Input feature dimension (PLIP features: 512)')
group.add_argument('--n_classes', default=2, type=int, help='Number of classes')
group.add_argument('--batch_size', default=1, type=int, help='Batch size')
group.add_argument('--max_patch_train', default=None, type=int, help='Maximum patches per training batch')
group.add_argument('--p_batch_size', default=512, type=int, help='Patch batch size')
group.add_argument('--p_batch_size_v', default=2048, type=int, help='Patch batch size for validation')
group.add_argument('--loss', default='ce', type=str, choices=['ce','bce','asl','nll_surv'], help='Loss function type')
group.add_argument('--opt', default='adam', type=str, help='Optimizer type [adam, adamw]')
group.add_argument('--model', default='e2e_r18_abmilx', type=str, help='Model name')
group.add_argument('--seed', default=2021, type=int, help='Random seed [2021]')
group.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate [0.0002]')
group.add_argument('--warmup_lr', default=1e-6, type=float, help='Warmup learning rate [1e-6]')
group.add_argument('--lr_sche', default='cosine', type=str, help='Learning rate scheduler [cosine, step, const]')
group.add_argument('--lr_supi', action='store_true', help='Update learning rate per iteration')
group.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [1e-5]')
group.add_argument('--accumulation_steps', default=1, type=int, help='Gradient accumulation steps')
group.add_argument('--clip_grad', default=None, type=float, help='Gradient clipping threshold')
group.add_argument('--always_test', action='store_true', help='Always test model during training')
group.add_argument('--best_metric_index', default=-1, type=int, help='Metric index for model selection')
group.add_argument('--model_ema', action='store_true', help='Enable Model EMA')
group.add_argument('--no_determ', action='store_true', help='Disable deterministic mode')
group.add_argument('--no_deter_algo', action='store_true', help='Disable deterministic algorithms')
group.add_argument('--deter_algo', action='store_true', help='Enable deterministic algorithms')
group.add_argument('--channels_last', action='store_true', default=False, help='Use channels_last memory layout')
group.add_argument('--sync_bn', action='store_true', default=False, help='Enable synchronized batch normalization')
group.add_argument('--reduce_bn', action='store_true', default=False, help='Enable reduced batch normalization')
group.add_argument('--no_drop_last', action='store_true', default=False, help='Disable dropping last incomplete batch')

##### Evaluation
group = parser.add_argument_group('Evaluation')
group.add_argument('--num_bootstrap', default=1000, type=int, help='Number of bootstrap iterations')
group.add_argument('--bootstrap_mode', default='test', type=str, choices=['test','none','val','test_val'])
group.add_argument('--bin_metric', action='store_true', help='Use binary average for binary classification')

##### Model
group = parser.add_argument_group('Model')
# General
group.add_argument('--act', default='relu', type=str, choices=['relu','gelu','none'], help='Activation function')
group.add_argument('--dropout', default=0.25, type=float, help='Dropout rate')
group.add_argument('--mil_norm', default=None, choices=['bn','ln','none'], help='MIL normalization type')
group.add_argument('--no_mil_bias', action='store_true', help='Disable MIL bias')
group.add_argument('--mil_bias', action='store_true', help='Enable MIL bias')
group.add_argument('--inner_dim', default=512, type=int, help='Inner dimension')
# Shuffle
group.add_argument('--patch_shuffle', action='store_true', help='Enable 2D patch shuffle')
# Common MIL models
group.add_argument('--da_act', default='relu', type=str, help='Activation function in DAttention')
group.add_argument('--da_gated', action='store_true', help='Enable gated DAttention')
# General Transformer
group.add_argument('--pos', default=None, type=str, choices=['ppeg','sincos','none',], help='Position encoding type')
group.add_argument('--n_heads', default=8, type=int, help='Number of attention heads')
group.add_argument('--n_layers', default=2, type=int, help='Number of transformer layers')
group.add_argument('--pool', default='cls_token', type=str, help='Pooling method')
group.add_argument('--attn_dropout', default=0., type=float, help='Attention dropout rate')
group.add_argument('--ffn', action='store_true', help='Enable FFN')
group.add_argument('--sdpa_type', default='torch', type=str, choices=['torch','flash','math','memo_effi','torch_math'], help='SDPA implementation type')
group.add_argument('--attn_type', default='sa', type=str, choices=['sa','ca'], help='Attention type')
group.add_argument('--ffn_dp', default=0., type=float, help='FFN dropout rate')
group.add_argument('--ffn_ratio', default=4., type=float, help='FFN expansion ratio')

##### E2E
group = parser.add_argument_group('E2E')
group.add_argument('--freeze_enc', action='store_true', help='Freeze encoder')
group.add_argument('--enc_init', default=None, type=str, help='Pretrained encoder weights path')
group.add_argument('--load_gpu_later', action='store_true', help='Load model to GPU later')
group.add_argument('--load_gpu_later_train', action='store_true', help='Load model to GPU later during training')
group.add_argument('--no_enc_pt', action='store_true', help='Disable encoder pretraining [Only for ablation]')
# Patch Selection
group.add_argument('--pad_enc_bs', action='store_true', help='Pad encoder batch size')
group.add_argument('--sel_type', default="random", type=str, choices=['random','ema'], help='Patch selection type [Only for ablation]')
group.add_argument('--num_group_1d', default=None, type=int, help='Number of 1D groups[Only for ablation]')
group.add_argument('--num_group_2d', default=None, type=int, help='Number of 2D groups[Only for ablation]')
group.add_argument('--all_patch_train', action='store_true', help='Train on all patches')
# Test
group.add_argument('--test_type', default='main', type=str, choices=['main','ema','both','both_ema'], help='Test mode type')

##### ABMILX
group.add_argument('--abx_D', default=None, type=float, help='ABX D ratio')
group.add_argument('--abx_attn_bias', action='store_true', help='Enable ABX attention bias')
group.add_argument('--abx_attn_plus', action='store_true', help='Enable ABX attention plus')
group.add_argument('--abx_pad_v', action='store_true', help='Enable ABX padding for values in attention plus')
group.add_argument('--abx_attn_plus_embed_new', action='store_true', help='Enable new ABX attention plus embedding')

##### RRT
group = parser.add_argument_group('RRT')
group.add_argument('--epeg_k', default=15, type=int, help='EPEG kernel size')
group.add_argument('--crmsa_k', default=3, type=int, help='CRMSA kernel size')
group.add_argument('--region_num', default=8, type=int, help='Number of regions')
group.add_argument('--rrt_n_heads', default=8, type=int, help='Number of RRT attention heads')
group.add_argument('--rrt_n_layers', default=2, type=int, help='Number of RRT layers')
group.add_argument('--rrt_pool', default="attn", type=str, help='RRT pooling type')

##### Miscellaneous
group = parser.add_argument_group('Miscellaneous')
group.add_argument('--title', default='default', type=str, help='Experiment title')
group.add_argument('--project', default='mil_new_c16', type=str, help='Project name')
group.add_argument('--log_iter', default=100, type=int, help='Logging frequency')
group.add_argument('--amp', action='store_true', help='Enable automatic mixed precision training')
group.add_argument('--amp_unscale', action='store_true', help='Enable AMP unscaling')
group.add_argument('--no_log', action='store_true', help='Disable logging')
group.add_argument('--output_path', type=str, help='Output path')
group.add_argument('--model_path', default=None, type=str, help='Model initialization path')
group.add_argument("--local-rank", "--local_rank", type=int)
group.add_argument('--script_mode', default='all', type=str, help='Script mode [all, no_train, test, only_train]')
group.add_argument('--profile', action='store_true', help='Enable profiling')
group.add_argument('--debug', action='store_true', help='Enable debug mode')
group.add_argument('--save_iter', default=-1, type=int, help='Save frequency')
group.add_argument('--mm', default=0.9997, type=float, help='Decay of EMA')
# Torchcompile
group.add_argument('--torchcompile', action='store_true', help='Enable torch.compile')
group.add_argument('--torchcompile_mode', default='default', type=str, choices=['default','reduce-overhead','max-autotune'], help='Torch compile mode')
# Wandb
group.add_argument('--wandb', action='store_true', help='Enable Weights & Biases')
group.add_argument('--wandb_watch', action='store_true', help='Enable Weights & Biases model watching')
# DDP
group.add_argument('--no_ddp_broad_buf', action='store_true', help='Disable DDP broadcast buffers')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()

    cfg = {}
    if args_config.config:
        config_files = args_config.config.split(',')

        for config_file in config_files:
            config_file = config_file.strip()  
            if config_file:  
                try:
                    with open(config_file, 'r') as f:
                        cfg.update(yaml.safe_load(f))
                except Exception as e:
                    print(f"Error loading config file {config_file}: {str(e)}")

        parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    args.config = args_config.config.split(',')

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    return args, args_text

def more_about_config(args):
    # Additional settings
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device = init_distributed_device(args)
    
    # Training settings
    if not args.mil_bias:
        args.mil_bias = not args.no_mil_bias
    args.drop_last = not args.no_drop_last
    args.prefetch = not args.no_prefetch
    
    # Debug settings
    if args.deter_algo:
        args.no_deter_algo = False 
    # Backup original seed for multi-fold random experiments
    args.seed_ori = args.seed

    # Assertions
    # Feature input currently does not support DDP training due to dataloader shuffle control
    if args.distributed and not args.image_input:
        raise NotImplementedError 

    if args.persistence:
        if args.same_psize > 0:
            raise NotImplementedError("Random same patch is different from not persistence")

    if args.val2test or args.val_ratio == 0.:
        args.always_test = False
    
    # EMA conflicts
    if args.model_ema and ('ema' in args.sel_type or args.test_type != 'main' or args.e2e_ema):
        raise NotImplementedError

    if args.datasets.lower() == 'panda':
        args.n_classes = 6

    if args.model in ('clam_sb', "clam_mb"):
        args.main_alpha = .7
        args.aux_alpha = .3
    elif args.model == 'dsmil':
        args.main_alpha = 0.5
        args.aux_alpha = 0.5

    if args.best_metric_index == -1:
        args.best_metric_index = 1 if args.n_classes != 2 and not args.datasets.lower().startswith('surv') else 0

    args.max_epoch = min(args.max_epoch,args.num_epoch)

    return args,device