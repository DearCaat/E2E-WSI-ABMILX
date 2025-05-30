import os
import torch
import numpy as np
import re
from torch.utils.data import Dataset
from .data_utils import *
from pathlib import Path

class FeatClsDataset(Dataset):
    def __init__(self, file_name=None, file_label=None,root=None,persistence=True,keep_same_psize=0,is_train=False,_type='nsclc',args=None):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(FeatClsDataset, self).__init__()

        self.patient_name = file_name
        self.slide_label = []
        self.root = root
        self.all_pts = os.listdir(os.path.join(self.root,'pt_files'))
        self.slide_name = []
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train
        self.same_psize_pad_type = args.same_psize_pad_type
        self.h5_path = args.h5_path

        for i,_patient_name in enumerate(self.patient_name):
            _sides = np.array([ _slide if _patient_name in _slide else '0' for _slide in self.all_pts])
            _ids = np.where(_sides != '0')[0]
            for _idx in _ids:
                if persistence:
                    _feat = torch.load(os.path.join(self.root,'pt_files',_sides[_idx]),weights_only=True)
                    if keep_same_psize:
                        _feat = get_smae_psize(_feat,keep_same_psize,args.same_psize_pad_type,args.min_seq_len)
                    self.slide_name.append(_feat)
                    self.slide_label.append(file_label[i])
                else:
                    self.slide_name.append(_sides[_idx])
                    self.slide_label.append(file_label[i])
        if _type.lower().startswith('bio'):
            self.slide_label = [int(_l) for _l in self.slide_label]
        else:
            if 'nsclc' in _type.lower():
                self.slide_label = [ 0 if _l == 'LUAD' else 1 for _l in self.slide_label]
            elif 'brca' in _type.lower():
                self.slide_label = [ 0 if _l == 'IDC' else 1 for _l in self.slide_label]
            elif 'call' in _type.lower():
                self.slide_label = [ 0 if _l == 'normal' else 1 for _l in self.slide_label]
            elif re.search(r'panda', _type.lower()) is not None:
                self.slide_label = [int(_l) for _l in self.slide_label]
            else:
                raise NotImplementedError

    def __len__(self):
        return len(self.slide_name)

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        file_path = self.slide_name[idx]
        label = self.slide_label[idx]

        if self.h5_path is not None:
            _pos = get_seq_pos_fn(os.path.join(self.h5_path,Path(file_path).stem+'.h5'))
        else:
            _pos = None

        if self.persistence:
            features = file_path
        else:
            features = torch.load(os.path.join(self.root,'pt_files',file_path),weights_only=True)
            if self.keep_same_psize:
                if _pos is not None:
                    features,_pos[1] = get_smae_psize(features,self.keep_same_psize,self.same_psize_pad_type,pos=_pos[1])
                else:
                    features = get_smae_psize(features,self.keep_same_psize,self.same_psize_pad_type)

        outputs = {'input': features, 'target':int(label)}

        if _pos is not None:
            _pos = torch.cat(_pos,dim=0)
            if (_pos.shape[0] - 1) != features.shape[0]:
                print(_pos.shape)
                print(features.shape)
                raise AssertionError
            outputs['pos'] = _pos

        return outputs

class FeatSurvDataset(Dataset):
    def __init__(self, df, root=None,persistence=True,keep_same_psize=0,is_train=False,args=None):
        self.root = os.path.join(root,'pt_files')
        # self.root = root
        self.persistence = persistence
        self.all_pts = os.listdir(self.root)
        self.keep_same_psize = keep_same_psize
        self.rows = df
        self.is_train = is_train

        self.same_psize_pad_type = args.same_psize_pad_type if args else 'pad'
        self.h5_path = args.h5_path if args else None

        # 得到case ID下的所有相关pt文件，用字典保存
        self.slide_name = {}
        for index, row in self.rows.iterrows():
            case_name = row['ID']
            if self.persistence:
                features = []
                patch_ids = []
                for slide_filename in self.all_pts:
                    if case_name in slide_filename:
                        feat = torch.load(os.path.join(self.root, slide_filename), weights_only=True)
                        pid = [f"{slide_filename[:-3]}-{i}" for i in range(feat.shape[0])]
                        features.append(feat)
                        patch_ids.extend(pid)

                if len(features) > 0:
                    features = torch.cat(features, dim=0)

                    if self.keep_same_psize and self.is_train:
                        features = get_smae_psize(features,self.keep_same_psize,self.same_psize_pad_type)

                    self.slide_name[case_name] = (features, patch_ids)

                else:
                    continue

            else:
                slides = [ slide for slide in self.all_pts if case_name in slide]
                
                if not slides:
                    continue
                
                self.slide_name[str(case_name)] = slides
        
        # 筛选有效样本，drop特征中不存在的row
        self.rows = self.rows[self.rows['ID'].apply(lambda x: x in self.slide_name and bool(self.slide_name[x]))]
        self.rows.reset_index(drop=True, inplace=True)  # 重新索引
        
    def read_WSI(self, path):
        wsi = []
        all_patch_id = []
        for x in path:
            _wsi = torch.load(os.path.join(self.root,x),weights_only=True)
            wsi.append(_wsi)
            all_patch_id += [str(x)[:-3]+'-'+str(i) for i in range(_wsi.shape[0])]
        wsi = torch.cat(wsi, dim=0)
        if self.keep_same_psize and self.is_train:
            wsi = get_smae_psize(wsi,self.keep_same_psize,self.same_psize_pad_type)
        return wsi,all_patch_id

    def __getitem__(self, index):
        case = self.rows.loc[index, ['Study', 'ID', 'Event', 'Status', 'Label']].values.tolist()
        Study, ID, Event, Status, Label = case
        Censorship = 1 if int(Status) == 0 else 0
        if self.persistence:
            WSI, all_patch_id = self.slide_name[str(ID)]
        else:
            WSI,all_patch_id = self.read_WSI(self.slide_name[ID])
            
        _pos = None
        if self.h5_path is not None:
            if isinstance(self.slide_name[str(ID)], str):
                h5_file_stem = Path(self.slide_name[str(ID)]).stem
            elif isinstance(self.slide_name[str(ID)], list) and len(self.slide_name[str(ID)]) == 1:
                h5_file_stem = Path(self.slide_name[str(ID)][0]).stem
            else:
                h5_file_stem = None

            if h5_file_stem is not None:
                pos_path = os.path.join(self.h5_path, h5_file_stem + '.h5')
                if os.path.isfile(pos_path):
                    _pos = get_seq_pos_fn(pos_path)
                    if self.keep_same_psize:
                        if _pos is not None:
                            WSI, _pos[1] = get_smae_psize(WSI,
                                                          self.keep_same_psize,
                                                          self.same_psize_pad_type,
                                                          pos=_pos[1])
                        else:
                            WSI = get_smae_psize(WSI,
                                                 self.keep_same_psize,
                                                 self.same_psize_pad_type)

        outputs = {
            'input': WSI,
            'event': Event,
            'censorship': Censorship,
            'target': Label 
        }

        if _pos is not None:
            _pos = torch.cat(_pos, dim=0)
            if (_pos.shape[0] - 1) != WSI.shape[0]:
                print(_pos.shape)
                print(WSI.shape)
                raise AssertionError("pos.shape 与特征.shape 不匹配")
            outputs['pos'] = _pos


        return outputs

    def __len__(self):
        return len(self.rows)

