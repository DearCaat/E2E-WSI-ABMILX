import pandas as pd
import re 
from PIL import Image
import os
import pickle
import io
import lmdb
import torch

from .data_utils import *

# one-level
class ClsLMDBDataset(Dataset):
    def __init__(self, args,env,df, keep_same_psize=0,mode="train",seed=2021,transforms=None,_type='nsclc',persistence=False,channels_last=False,img_size=224,h5_root=None):
        super(ClsLMDBDataset, self).__init__()
        self.img_size = img_size
        self.rows = df
        self.channels_last = channels_last
        if type(env) == str:
            self.lmdb_path = env
            self.env = self.open_lmdb()
        else:
            self.env = env
        self.seed = seed
        self.keep_same_psize = keep_same_psize
        self.mode = mode
        self.transforms = transforms
        self.args = args
        self.h5_root = h5_root
        self.current_epoch = 0

        with self.env.begin(write=False) as txn:
            # key1_sub1, key1_sub2, key2_sub1, key2_sub2,...
            self.PN_dict = pickle.loads(txn.get(b'__pn__'))
            self.all_slides = pickle.loads(txn.get(b'__slide__'))

        if hasattr(self,'lmdb_path'):
            self.env = self.close_lmdb()
        # Create a boolean mask to mark rows to keep
        new_rows = []
        for idx, row in self.rows.iterrows():
            current_id = row['ID']
            # Condition 1: Exact match
            if current_id in self.all_slides:
                new_rows.append(row)
                continue

            # Condition 2: Prefix match
            matched_slides = [slide for slide in self.all_slides if slide.startswith(current_id)]

            # If there are prefix matches
            if matched_slides:
                # Create a new row for each matched slide
                for matched_slide in matched_slides:
                    new_row = row.copy()  # Copy all data from original row
                    new_row['ID'] = matched_slide  # Update ID to matched slide
                    new_rows.append(new_row)

        # Convert all kept rows to new DataFrame
        self.rows = pd.DataFrame(new_rows)
        self.rows.reset_index(drop=True, inplace=True)

        # Get all related image files under case ID, store in dictionary
        self.all_imgs = {}
        # Always keep complete patch information without dropping patches
        self.all_imgs_all = {}
        self.all_pos = {}

        for i, row in self.rows.iterrows():
            _slide = row['ID']
            _slide_specific_keys = [str(_slide)+'-'+str(i) for i in range(self.PN_dict[_slide])]
            if self.h5_root is not None:
                _seq_pos = get_seq_pos_fn(os.path.join(self.h5_root,_slide+'.h5'))
                
            self.all_imgs_all[str(_slide)] = _slide_specific_keys
            self.all_imgs[str(_slide)] = _slide_specific_keys
            if self.h5_root is not None:
                self.all_pos[str(_slide)] = torch.cat(_seq_pos,dim=0)

        if mode == 'train':
            args.all_imgs_train = self.all_imgs
            args.all_imgs_all_train = self.all_imgs_all
        elif mode == 'val':
            args.all_imgs_val = self.all_imgs
        elif mode == 'test':
            args.all_imgs_test = self.all_imgs

        # label
        if _type.lower().startswith('bio'):
            self.rows.loc[:, 'Label'] = self.rows['Label'].astype(int)
        else:
            if 'nsclc' in _type.lower():
                self.rows.loc[:, 'Label'] = self.rows['Label'].map({'LUAD': 0, 'LUSC': 1})
            elif 'brca' in _type.lower():
                self.rows.loc[:, 'Label'] = self.rows['Label'].map({'IDC': 0, 'ILC': 1})
            elif 'call' in _type.lower():
                self.rows.loc[:, 'Label'] = self.rows['Label'].map({'normal': 0, 'tumor': 1})
            elif re.search(r'panda', _type.lower()) is not None:
                self.rows.loc[:, 'Label'] = self.rows['Label'].astype(int)
                pass
            else:
                raise NotImplementedError
    
    def set_epoch(self,epoch):
        self.current_epoch = epoch

    def open_lmdb(self):

        GB = 1024 * 1024 * 1024  # 1 GB in bytes
        
        self.env = lmdb.open(self.lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, map_size=100*GB)

        return self.env
    
    def update_imgs(self):
        if self.mode == 'train':
            self.all_imgs = self.args.all_imgs_train
        elif self.mode == 'val':
            self.all_imgs = self.args.all_imgs_val
        elif self.mode == 'test':
            self.all_imgs = self.args.all_imgs_test
        
    def close_lmdb(self):
        if self.env is not None:
            self.env.close()
        return None
    def load_data(self,slide_name,valid_imgs,length,keys):
        imgs = {}
        with self.env.begin(write=False, buffers=True) as txn:
            cursor = txn.cursor()
            cursor.set_key(u"{}-0".format(slide_name).encode('ascii'))
            for i, (_key, _value) in enumerate(cursor.iternext()):
                if i == length:
                    break
                img_data = pickle.loads(_value.tobytes())
                imgs[keys[i]] = Image.open(io.BytesIO(img_data))

        imgs = dict(sorted(imgs.items(),key=lambda x: x[0],reverse=False))

        return list(imgs.values())
    
    def load_data_random(self,valid_imgs):
        if self.channels_last:
            imgs = torch.empty((len(valid_imgs), 3, self.img_size, self.img_size),memory_format=torch.channels_last)
        else:
            imgs = torch.empty((len(valid_imgs), 3,self.img_size, self.img_size))
        with self.env.begin(write=False, buffers=True) as txn:
            for i,key_str in enumerate(valid_imgs):
                imgs[i] = torch.from_numpy(imfrombytes(pickle.loads(txn.get(key_str.encode('ascii')).tobytes())).transpose(2, 0, 1))

        return imgs
    
    def __del__(self):
        if hasattr(self,'lmdb_path'):
            self.close_lmdb()

    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        slide_id = row['ID']
        label = row['Label']
        if self.env is None:
            self.env = self.open_lmdb()

        if self.h5_root:
            _pos = self.all_pos[slide_id]

        if self.keep_same_psize > 0.:
            _all_imgs = get_same_psize_img(self.args,self.all_imgs_all[slide_id],self.keep_same_psize,self.all_imgs[slide_id])
            
            if self.h5_root:
                _selected_indices = [int(key.split('-')[-1]) for key in _all_imgs]
                _pos = torch.cat([_pos[0].unsqueeze(0),_pos[1:][_selected_indices]],dim=0)
        else:
            _all_imgs = self.all_imgs[slide_id]

        imgs = self.load_data_random(_all_imgs)

        outputs = {'input': imgs, 'target': torch.tensor(label).unsqueeze(0)}
        
        if self.h5_root:
            outputs['pos'] = _pos
      
        return outputs

class SurvLMDBDataset(Dataset):
    def __init__(self, args, env, df, keep_same_psize=0, mode="train", seed=2021, transforms=None, persistence=False, channels_last=False, img_size=224, h5_root=None):
        super(SurvLMDBDataset, self).__init__()

        if h5_root is not None:
            raise NotImplementedError

        self.rows = df
        self.img_size = img_size
        self.channels_last = channels_last
        if isinstance(env, str):
            self.lmdb_path = env
            self.env = self.open_lmdb()
        else:
            self.env = env
            
        self.seed = seed
        self.keep_same_psize = keep_same_psize
        self.mode = mode
        self.transforms = transforms
        self.args = args
        self.h5_root = h5_root
        self.current_epoch = 0

        with self.env.begin(write=False) as txn:
            self.PN_dict = pickle.loads(txn.get(b'__pn__'))
            self.all_slides = pickle.loads(txn.get(b'__slide__'))

        if hasattr(self,'lmdb_path'):
            self.env = self.close_lmdb()
            
        self.slide_name = {}
        self.all_imgs = {}
        self.all_imgs_all = {}
        self.all_pos = {}

        for index, row in self.rows.iterrows():
            case_name = row['ID']
            slides = [slide for slide in self.all_slides if case_name in slide]

            self.slide_name[str(case_name)] = slides
            
            _slide_specific_keys = []
            _seq_pos = []
            for _slide in slides:
                _slide_specific_keys += [str(_slide)+'-'+str(i) for i in range(self.PN_dict[_slide])]

            self.all_imgs_all[str(case_name)] = _slide_specific_keys
            self.all_imgs[str(case_name)] = _slide_specific_keys
    
        self.rows = self.rows[self.rows['ID'].apply(lambda x: x in self.slide_name and bool(self.slide_name[x]))]
        self.rows.reset_index(drop=True, inplace=True)

        if mode == 'train':
            args.all_imgs_train = self.all_imgs
            args.all_imgs_all_train = self.all_imgs_all
        elif mode == 'val':
            args.all_imgs_val = self.all_imgs
        elif mode == 'test':
            args.all_imgs_test = self.all_imgs

    def set_epoch(self,epoch):
        self.current_epoch = epoch

    def open_lmdb(self):
        GB = 1024 * 1024 * 1024
        return lmdb.open(self.lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, map_size=100*GB)
    
    def update_imgs(self):
        if self.mode == 'train':
            self.all_imgs = self.args.all_imgs_train
        elif self.mode == 'val':
            self.all_imgs = self.args.all_imgs_val
        elif self.mode == 'test':
            self.all_imgs = self.args.all_imgs_test

    def close_lmdb(self):
        if self.env is not None:
            self.env.close()
        return None

    def load_data_random(self,valid_imgs):
        if self.channels_last:
            imgs = torch.empty((len(valid_imgs), 3, self.img_size, self.img_size),memory_format=torch.channels_last)
        else:
            imgs = torch.empty((len(valid_imgs), 3,self.img_size, self.img_size))
        with self.env.begin(write=False, buffers=True) as txn:
            try:
                for i,key_str in enumerate(valid_imgs):
                    imgs[i] = torch.from_numpy(imfrombytes(pickle.loads(txn.get(key_str.encode('ascii')).tobytes())).transpose(2, 0, 1))
            except:
                print(key_str)
        return imgs

    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        Study, case_name, Event, Status, label = self.rows.loc[idx, ['Study', 'ID', 'Event', 'Status', 'Label']].values.tolist()
        Censorship = 1 if int(Status) == 0 else 0

        if self.env is None:
            self.env = self.open_lmdb()

        if self.h5_root:
            _pos = self.all_pos[case_name]


        if self.keep_same_psize > 0.:
            _all_imgs = get_same_psize_img(self.args,self.all_imgs_all[case_name],self.keep_same_psize,self.all_imgs[case_name])
            
            if self.h5_root:
                _selected_indices = [int(key.split('-')[-1]) for key in _all_imgs]
                _pos = torch.cat([_pos[0].unsqueeze(0),_pos[1:][_selected_indices]],dim=0)
        else:
            _all_imgs = self.all_imgs[case_name]

        imgs = self.load_data_random(_all_imgs)

        outputs = {
            "input": imgs,
            "event": torch.tensor(Event).unsqueeze(0),
            "censorship": torch.tensor(Censorship).unsqueeze(0),
            "target": torch.tensor(label).unsqueeze(0)
        }

        if self.h5_root:
            outputs['pos'] = _pos
            
        return outputs

    def __del__(self):
        if hasattr(self, 'lmdb_path'):
            self.close_lmdb()