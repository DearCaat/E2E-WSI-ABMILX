import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from timm.layers.helpers import to_2tuple
# from utils.loss import loss_fg,loss_bg,SupConLoss
# from utils.memory import Memory
#
# from transformers import CLIPTokenizer, CLIPTextModel

class ConvStem(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=False, **kwargs):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        else:
            x = x.permute(0,2,3,1)
        x = self.norm(x)
        return x

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class Att_Head(nn.Module):
    def __init__(self,FEATURE_DIM,ATT_IM_DIM):
        super(Att_Head, self).__init__()

        self.fc1 = nn.Linear(FEATURE_DIM, ATT_IM_DIM)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ATT_IM_DIM, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x 1, N * D
class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


# clip_tokenizer = CLIPTokenizer.from_pretrained()
#
# clip_tokenizer = CLIPTokenizer.from_pretrained('./')
#
# text_encoder = CLIPTextModel.from_pretrained(
#     './',
#     subfolder="text_encoder")



class CHIEF(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=True, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(),dataset=None,**kwargs):
        super(CHIEF, self).__init__()
        self.size_dict = {'xs': [384, 256, 256], "small": [768, 512, 256], "big": [1024, 512, 384], 'large': [2048, 1024, 512]}
        size = self.size_dict[size_arg]

        if 'brca' in dataset:
            self.anatomic_index = 1
        elif 'blca' in dataset:
            self.anatomic_index = 2
        elif 'panda' in dataset:
            self.anatomic_index = 4
        elif 'luad' in dataset or 'lusc' in dataset or 'nsclc' in dataset:
            self.anatomic_index = 6
        else:
            raise ValueError('Dataset not found')

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        initialize_weights(self)

        self.att_head = Att_Head(size[1],size[2])
        self.text_to_vision=nn.Sequential(nn.Linear(768, size[1]), nn.ReLU(), nn.Dropout(p=0.25))

        self.register_buffer('organ_embedding', torch.randn(19, 768))

        if 'CHIEF_TEXT_PATH' not in os.environ or not os.environ['CHIEF_TEXT_PATH']:
            os.environ['CHIEF_TEXT_PATH'] = '/XXXwsi_data/ckp/chief/Text_emdding.pth'
        if os.path.exists(os.environ['CHIEF_TEXT_PATH']):
            word_embedding = torch.load(os.environ['CHIEF_TEXT_PATH'])
        else:
            raise ValueError('Text embedding not found')

        self.organ_embedding.data = word_embedding.float()
        self.text_to_vision=nn.Sequential(nn.Linear(768, size[1]), nn.ReLU(), nn.Dropout(p=0.25))

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    def forward(self, h,**kwargs):
        batch = self.anatomic_index
        #h_ori = h
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        #A_raw = A
        A = F.softmax(A, dim=1)

        embed_batch = self.organ_embedding[batch]
        embed_batch=self.text_to_vision(embed_batch)
        WSI_feature = torch.mm(A, h)
        #slide_embeddings = torch.mm(A, h_ori)

        M = WSI_feature+embed_batch

        logits = self.classifiers(M)

        # result = {
        #     'bag_logits': logits,
        #     'attention_raw': A_raw,
        #     'WSI_feature': slide_embeddings,
        #     'WSI_feature_anatomical': M
        # }
        return logits


    def patch_probs(self, h,x_anatomic):
        batch = x_anatomic
        A, h = self.attention_net(h)
        A_raw = A
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)

        embed_batch = self.organ_embedding[batch]
        embed_batch=self.text_to_vision(embed_batch)
        M = torch.mm(A, h)
        M = M+embed_batch
        bag_logits = self.classifiers(M)
        bag_prob = torch.softmax(bag_logits.squeeze(), dim=0)
        patch_logits = self.classifiers(h+embed_batch)
        patch_prob = torch.sigmoid(A_raw.squeeze()) * torch.softmax(patch_logits, dim=1)[:, 1]

        return{
            'bag_prob': bag_prob,
            'patch_prob': patch_prob,
            'attention_raw': A_raw.squeeze()
        }
