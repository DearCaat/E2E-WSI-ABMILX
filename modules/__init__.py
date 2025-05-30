import torch
import timm

from .abmil import DAttention,AttentionGated
from .abmilx import DAttentionX
from .clam import CLAM_MB,CLAM_SB
from .dsmil import MILNet
from .transmil import TransMIL
from .mean_max import MeanMIL,MaxMIL
from .dtfd import DTFD
from .rrt import RRTMIL
from .e2e import E2E
from .encoders import ResNetEncoder
from .e2e_pooling import MeanPooling,Attention
from .vit_mil import ViTMIL
from .gigap import GIGAPMIL
from .chief import CHIEF,ConvStem
from .utils import get_mil_model_params

import os 
from utils import ModelEmaV3


def load_enc_ckp(args,enc,enc_init_path):
    try:
        enc_ckp = torch.load(enc_init_path,weights_only=True,map_location='cpu')
    except:
        enc_ckp = torch.load(enc_init_path,map_location='cpu')
    
    if 'state_dict' in enc_ckp:
        enc_ckp = enc_ckp['state_dict']

    if 'model' in enc_ckp:
        enc_ckp = enc_ckp['model']
    new_state_dict = {}
    for key in enc_ckp:
        # 将 'classifier.0.weight' 改为 'classifier.weight'
        if 'encoder.' in key:
            #if not '.num_batches_tracked' in key:
            new_key = key.replace('encoder.', '')
            if 'module.' in new_key:
                new_key = new_key.replace('module.', '')
        # torchvision ckp
        elif 'resnet.' in key:
            new_key = key.replace('resnet.', '')
        else:
            new_key = key
        # 新创建的模型这里是空，暂且不知道原因
        if not 'model.layer3.1.bn2.num_batches_tracked' in new_key:
            new_state_dict[new_key] = enc_ckp[key]
    enc_ckp = new_state_dict
    info = enc.load_state_dict(enc_ckp,strict=False)

    if args.rank == 0:
        print(f"Enc Loading: {enc_init_path}")
        print(f"Results: {info}")
    
    return enc

def build_model(args,device,train_loader):
    if args.image_input:
        others = {}
        enc_name,mil_name = args.model.split('_',3)[1],args.model.split('_',3)[2]
        encoder = build_encoder(args,enc_name)
        if args.enc_init is not None:
            if os.path.isdir(args.enc_init):
                enc_init_path = os.path.join(args.enc_init,f"fold_{args.fold_curr}_model_best.pt")
            else:
                enc_init_path = args.enc_init

            encoder = load_enc_ckp(args,encoder,enc_init_path)

        mil,others = build_mil(args,mil_name,device,train_loader)

        model = E2E(encoder,mil,device,args).to(device)

        if 'ema' in args.sel_type \
        or args.test_type != 'main':
            model_tea = ModelEmaV3(model,decay=args.mm)
            
            others['model_ema'] = model_tea

        return model,others
    else:
        return build_mil(args,args.model,device,train_loader)

def build_encoder(args,model_name):
    print('loading model checkpoint')
    if model_name == 'r50':
        model = ResNetEncoder(pretrained=not args.no_enc_pt)
    elif model_name == 'r50v2':
        model = ResNetEncoder('resnet50.tv2_in1k',pretrained=not args.no_enc_pt)
    elif model_name == 'r18':
        model = ResNetEncoder('resnet18.tv_in1k',pretrained=not args.no_enc_pt)
    elif model_name == 'r18a':
        kwargs = {'features_only': True, 'pretrained': True, 'num_classes': 0,'out_indices': (4,)}
        model = ResNetEncoder('resnet18.tv_in1k',kwargs=kwargs,pretrained=not args.no_enc_pt)
    elif model_name == 'uni':
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'chief':
        model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False,num_classes=0)
    elif model_name == 'gigap':
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False,num_classes=0)
    else:
        raise NotImplementedError(f'{model_name} not implemented')
    return model

def build_mil(args,model_name,device,train_loader):
    others = {}

    genera_model_params,genera_trans_params = get_mil_model_params(args)

    if model_name == 'rrtmil':
        model = RRTMIL(epeg_k=args.epeg_k,crmsa_k=args.crmsa_k,region_num=args.region_num,n_heads=args.rrt_n_heads,n_layers=args.rrt_n_layers,**genera_model_params).to(device)
    elif model_name == 'abmil':
        model = DAttention(**genera_model_params).to(device)
    elif model_name == 'abmilx':
        _ = genera_trans_params.pop('attn_type')
        model = DAttentionX(
            **genera_trans_params,
            D = args.abx_D,
            attn_type = 'mlp',
            attn_bias = args.abx_attn_bias,
            attn_plus = args.abx_attn_plus,
            pad_v = args.abx_pad_v,
            attn_plus_embed_new=args.abx_attn_plus_embed_new,
            ).to(device)
    elif model_name == 'gabmil':
        model = AttentionGated(**genera_model_params).to(device)
    # follow the official code
    # ref: https://github.com/mahmoodlab/CLAM
    elif model_name == 'clam_sb':
        model = CLAM_SB(**genera_model_params).to(device)
    elif model_name == 'clam_mb':
        model = CLAM_MB(**genera_model_params).to(device)
    elif model_name == 'transmil':
        model = TransMIL(**genera_trans_params).to(device)
    elif model_name == 'vitmil':
        model = ViTMIL(**genera_trans_params).to(device)
    elif model_name == 'dsmil':
        model = MILNet(**genera_model_params).to(device)
        if args.aux_alpha == 0.:
            args.main_alpha = 0.5
            args.aux_alpha = 0.5
    elif model_name == 'dtfd':
        model = DTFD(device=device, lr=args.lr, weight_decay=args.weight_decay, steps=args.num_epoch, input_dim=args.input_dim, n_classes=args.n_classes).to(device)
    elif model_name == 'meanmil':
        model = MeanMIL(**genera_model_params).to(device)
    elif model_name == 'maxmil':
        model = MaxMIL(**genera_model_params).to(device)
    elif model_name == 'meanP':
        model = MeanPooling(**genera_model_params).to(device)
    elif model_name == 'attn':
        model = Attention(**genera_model_params).to(device)
    elif model_name == 'gigap':
        model = GIGAPMIL(**genera_model_params).to(device)
    elif model_name == 'chief':
        model = CHIEF(**genera_model_params,dataset=args.datasets.lower()).to(device)
        if 'CHIEF_MIL_PATH' not in os.environ or not os.environ['CHIEF_MIL_PATH']:
            os.environ['CHIEF_MIL_PATH'] = '/XXXwsi_data/ckp/chief/CHIEF_pretraining.pth'
        if os.path.exists(os.environ['CHIEF_MIL_PATH']):
            state_dict = torch.load(os.environ['CHIEF_MIL_PATH'], map_location="cpu")["model"]
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if len(missing_keys) > 0:
                for k in missing_keys:
                    print("Missing ", k)

            if len(unexpected_keys) > 0:
                for k in unexpected_keys:
                    print("Unexpected ", k)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    return model, others