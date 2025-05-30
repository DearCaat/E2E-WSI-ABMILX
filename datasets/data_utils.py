from sklearn.model_selection import StratifiedKFold
import pandas as pd
import torch
import torch.nn.functional as F
import math
import h5py
from random import shuffle
from torchaug import transforms as ta_transforms
from collections import defaultdict
import pickle
from typing import Optional
import numpy as np
from turbojpeg import TJCS_RGB, TJPF_BGR, TJPF_GRAY, TurboJPEG
from contextlib import suppress
from functools import partial
from torch.utils.data import Dataset
import random
from copy import deepcopy

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]

MODEL2CONSTANTS = {
    "resnet50_trunc": {
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD
    },
    "uni_v1":
    {
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD
    },
    "conch_v1":
    {
        "mean": OPENAI_MEAN,
        "std": OPENAI_STD
    }
}

def check_tensor(tensor, tensor_name=""):
    if torch.isnan(tensor).any():
        print(f"{tensor_name} contains NaN values")
        raise ValueError
    if torch.isinf(tensor).any():
        print(f"{tensor_name} contains Inf values")
        raise ValueError
    if torch.isfinite(tensor).all():
        pass
        #print(f"{tensor_name} contains only finite values")  
        #raise ValueError
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

class PrefetchLoader:
    def __init__(
            self,
            loader,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            channels=3,
            device=torch.device('cuda'),
            img_dtype=torch.float32,
            need_norm=True,
            need_transform=False,
            transform_type='strong',
            img_size=224,
            is_train=True,
            trans_chunk=4,
            crop_scale=0.08,
            load_gpu_later=False):

        normalization_shape = (1, channels, 1, 1)

        self.loader = loader
        self.need_norm = need_norm

        if need_transform:
            if is_train:
                if transform_type == 'strong':
                    ta_list = [
                        ta_transforms.RandomHorizontalFlip(p=0.5),
                        ta_transforms.RandomVerticalFlip(p=0.5),
                    ]
                elif transform_type == 'strong_v2':
                    ta_list = [
                        ta_transforms.RandomHorizontalFlip(p=0.5),
                        ta_transforms.RandomVerticalFlip(p=0.5),
                        ta_transforms.RandomAffine(degrees=0, translate=(0.5, 0.5)),
                    ]
                elif transform_type == 'weak_strong':
                    ta_list = [
                        ta_transforms.RandomAffine(degrees=0, translate=(100 / 256, 100 / 256)),
                    ]
                else:
                    ta_list = [
                    ]
                if img_size != 224:
                    ta_list += [
                        ta_transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
                    ]
                else:
                    ta_list += [
                        ta_transforms.Resize(224),
                    ]
                self.transform = ta_transforms.SequentialTransform(
                    ta_list,
                    inplace=True,
                    batch_inplace=True,
                    batch_transform=True,
                    num_chunks=trans_chunk,
                    permute_chunks=False,
                )
            else:
                self.transform = ta_transforms.SequentialTransform(
                    [
                        ta_transforms.CenterCrop(224),
                    ],
                    inplace=True,
                    batch_inplace=True,
                    batch_transform=True,
                    num_chunks=1,
                    permute_chunks=False,
                )
        else:
            self.transform = None
        self.device = self.device_input = device
        self.img_dtype = img_dtype
        self.no_gpu_key = []
        if load_gpu_later:
            device = torch.device('cpu')
            self.device_input=device
            self.no_gpu_key = ['input']
        
        self.mean = torch.tensor(
            [x * 255 for x in mean], device=device, dtype=img_dtype).view(normalization_shape)
        self.std = torch.tensor(
            [x * 255 for x in std], device=device, dtype=img_dtype).view(normalization_shape)

        self.is_cuda = torch.cuda.is_available() and device.type == 'cuda'

    def __iter__(self):
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = suppress

        for next_batch in self.loader:
            with stream_context():
                next_batch = {
                    k: ([v.to(device=self.device_input, non_blocking=True) for v in v] if isinstance(v, list) and k == 'input' else 
                        v.to(device=self.device, non_blocking=True) if isinstance(v, torch.Tensor) and k not in self.no_gpu_key else v)
                    for k, v in next_batch.items()
                }
                if self.transform is not None:
                    if isinstance(next_batch['input'], list):
                        next_batch['input'] = [self.transform(tensor) for tensor in next_batch['input']]
                    else:
                        next_batch['input'] = self.transform(next_batch['input'])
                if self.need_norm:
                    if isinstance(next_batch['input'], list):
                        next_batch['input'] = [tensor.to(self.img_dtype).sub_(self.mean).div_(self.std) for tensor in next_batch['input']]
                    else:
                        next_batch['input'] = next_batch['input'].to(self.img_dtype).sub_(self.mean).div_(self.std)

            if not first:
                yield batch
            else:
                first = False

            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)

            batch = next_batch

        yield batch

    def __len__(self):
        return len(self.loader)
    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

def _jpegflag(flag: str = 'color', channel_order: str = 'bgr'):
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'color':
        if channel_order == 'bgr':
            return TJPF_BGR
        elif channel_order == 'rgb':
            return TJCS_RGB
    elif flag == 'grayscale':
        return TJPF_GRAY
    else:
        raise ValueError('flag must be "color" or "grayscale"')


global jpeg
jpeg = TurboJPEG()

def imfrombytes(content: bytes,
                flag: str = 'color',
                channel_order: str = 'bgr',
                backend: Optional[str] = None) -> np.ndarray:
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Same as :func:`imread`.
        channel_order (str): The channel order of the output, candidates
            are 'bgr' and 'rgb'. Default to 'bgr'.
        backend (str | None): The image decoding backend type. Options are
            `cv2`, `pillow`, `turbojpeg`, `tifffile`, `None`. If backend is
            None, the global imread_backend specified by ``mmcv.use_backend()`
            will be used. Default: None.

    Returns:
        ndarray: Loaded image array.

    Examples:
        >>> img_path = '/path/to/img.jpg'
        >>> with open(img_path, 'rb') as f:
        >>>     img_buff = f.read()
        >>> img = mmcv.imfrombytes(img_buff)
        >>> img = mmcv.imfrombytes(img_buff, flag='color', channel_order='rgb')
        >>> img = mmcv.imfrombytes(img_buff, backend='pillow')
        >>> img = mmcv.imfrombytes(img_buff, backend='cv2')
    """

    img = jpeg.decode(content, _jpegflag(flag, channel_order))
    if img.shape[-1] == 1:
        img = img[:, :, 0]
    return img

def lexicalOrder(n: int):
        ans = [0] * n
        num = 0
        for i in range(n):
            ans[i] = num
            if i == 0: 
                num = 1
                continue
            if num * 10 <= n:
                num *= 10
            else:
                while num % 10 == 9 or num + 1 > n:
                    num //= 10
                num += 1
        return ans

def get_seq_pos_fn(h5_path):
    h5_file = h5py.File(h5_path)

    img_x,img_y = h5_file['coords'].attrs['level_dim'] * h5_file['coords'].attrs['downsample']
    patch_size = h5_file['coords'].attrs['patch_size'] * h5_file['coords'].attrs['downsample']
    img_x,img_y = int(img_x),int(img_y)
    patch_size = [int(patch_size[0]),int(patch_size[1])]
    pos = []
    pw = img_x // patch_size[0]
    ph = img_y // patch_size[1]

    for _coord in np.array(h5_file['coords']):
        # Calculate current patch's 2D position in the large image
        patch_x = _coord[0] // patch_size[0]  # Horizontal position
        patch_y = _coord[1] // patch_size[1]  # Vertical position

        assert patch_x >= 0 and patch_y >= 0
        if patch_x >= pw:
            pw += 1
        if patch_y >= ph:
            ph += 1

        assert patch_x < pw and patch_y < ph

        pos.append([patch_x,patch_y])

    pos = torch.tensor(np.array(pos,dtype=np.int64))
    pos_all = torch.tensor(np.array([[pw,ph]],dtype=np.int64))
    try:
        check_tensor(pos)
        check_tensor(pos_all)
    except:
        with open('../tmp/log', 'a') as f:
            [print(_pos,file=f) for _pos in pos]
            #print(pos, file=f)
        print(h5_path)
        print(img_x,img_y)
        print(patch_size)
        print(pos)
        print(pos_all)
        assert 1 == 2

    return [pos_all,pos]

def split_by_key(input_list):
    result = defaultdict(list)
    for item in input_list:
        key, sub = item.rsplit('-', 1)
        result[key].append(item)
    return dict(result)


def load_lmdb(env):
    data = {}
    imgs = {}
    with env.begin(write=False,buffers=True) as txn:
        data['PN_dict'] = pickle.loads(txn.get(b'__pn__'))
        data['all_slides'] =  pickle.loads(txn.get(b'__slide__'))

        cursor = txn.cursor()
        for _slide in data['all_slides']:
            raw_data = {}
            _keys = lexicalOrder(data['PN_dict'][_slide])
            cursor.set_key(u"{}-0".format(_slide).encode('ascii'))
            for i,(key, value) in enumerate(cursor):
                if i == data['PN_dict'][_slide]:
                    break
                raw_data[_keys[i]] = value.tobytes()

            raw_data = dict(sorted(raw_data.items(),key=lambda x: x[0],reverse=False))
            imgs[_slide] = raw_data.values()

        data['imgs'] = imgs

    env.close()

    return data

def get_same_psize_img(args,_list,same_psize,list_random=None):
    assert same_psize > 0

    if same_psize > 0 and same_psize < 1:
        same_psize = same_psize * len(_list)
        same_psize = max(same_psize,args.min_seq_len)
    same_psize = int(same_psize)
    __list = deepcopy(_list)
    ps = len(__list)

    if ps > same_psize:
        if args.num_group_1d is None and args.num_group_2d is None:
            if list_random is not None and 0 < args.same_psize_ratio < 1:
                num_replace = int(same_psize * args.same_psize_ratio)
                # 创建一个不包含 list_random 中元素的新列表
                unique_elements = list(set(__list) - set(list_random))
                #if len(unique_elements) > 0:
                if len(unique_elements) >= num_replace:
                    replace_items = random.sample(unique_elements, num_replace)
                    replace_indices = random.sample(range(same_psize), num_replace)

                    for idx, item in zip(replace_indices, replace_items):
                        list_random[idx] = item
                    return list_random
            shuffle(__list)
            return __list[:same_psize]
        elif args.num_group_1d is not None:
            # 计算每个区间的大小
            interval_size = math.ceil(ps / args.num_group_1d)

            # 生成所有区间的起始索引
            interval_starts = np.arange(0, ps, interval_size)
            num_actual_intervals = len(interval_starts)

            # 为每个区间生成一个随机偏移
            random_offsets = np.random.randint(0, interval_size, size=num_actual_intervals)
            # 确保不会超出列表范围
            random_offsets = np.minimum(random_offsets, ps - interval_starts - 1)

            # 计算最终的索引
            indices = (interval_starts + random_offsets)[:same_psize]

            # 如果区间采样数量不足，补充随机索引
            if len(indices) < same_psize:
                # 生成所有可能的索引
                all_indices = np.arange(ps)
                # 移除已经选择的索引
                mask = np.ones(ps, dtype=bool)
                mask[indices] = False
                remaining_indices = all_indices[mask]
                # 随机选择补充的索引
                np.random.shuffle(remaining_indices)
                additional_indices = remaining_indices[:same_psize - len(indices)]
                indices = np.concatenate([indices, additional_indices])

            return [_list[i] for i in indices]
        elif args.num_group_2d is not None:
            return _get_same_psize_img_region(__list,same_psize,args.num_group_2d)
    elif ps < same_psize and args.pad_enc_bs and args.batch_size > 1:
        random_indices = torch.randint(0, ps, (same_psize - ps,))
        padding = [__list[pad_idx] for pad_idx in random_indices]
        __list += padding
        #print(len(_list))
    return __list

def _get_same_psize_img_region(_list, same_psize, grid_num=None):
    """
    Reshape a 1D list into a 2D grid and uniformly sample from local regions

    Parameters:
    - _list: Input 1D list
    - same_psize: Total number of samples required
    - grid_num: Number of regions to divide along each dimension

    Returns:
    - List of sampled items with length exactly equal to same_psize
    """
    ps = len(_list)

    # Create a resizable dataset to hold the output
    img_shape = img_patch.shape
    # Calculate the closest square size
    side_length = int(math.ceil(math.sqrt(ps)))
    # Pad the list to square shape
    padded_list = _list + [_list[-1]] * (side_length * side_length - ps)
    # Reshape to 2D array
    array_2d = np.array(padded_list).reshape(side_length, side_length)

    # Calculate grid dimensions
    if grid_num is None:
        grid_num = 3  # Default value

    grid_size = (side_length // grid_num, side_length // grid_num)
    total_cells = grid_num * grid_num

    # Calculate samples per cell, ensuring exact total
    samples_per_cell = same_psize // total_cells
    remaining_samples = same_psize % total_cells

    selected_indices = []
    remaining_indices = []

    # Sample from each region
    for i in range(grid_num):
        for j in range(grid_num):
            # Calculate current region boundaries
            row_start = i * grid_size[0]
            row_end = min((i + 1) * grid_size[0], side_length)
            col_start = j * grid_size[1]
            col_end = min((j + 1) * grid_size[1], side_length)

            # Get all indices from current region (using array_2d)
            cell_indices = []
            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    index = r * side_length + c
                    if index < ps:  # Ensure index is within original list range
                        cell_indices.append(index)

            # Random sampling
            if cell_indices:
                np.random.shuffle(cell_indices)
                if len(cell_indices) <= samples_per_cell:
                    # If not enough indices in this cell, take all of them
                    selected_indices.extend(cell_indices)
                    # Track deficit for later compensation
                    remaining_indices.extend([i for i in range(ps) if i not in selected_indices])
                else:
                    selected_indices.extend(cell_indices[:samples_per_cell])
                    # Add remaining indices to the pool for potential additional sampling
                    remaining_indices.extend(cell_indices[samples_per_cell:])

    # Ensure exact number of samples
    if len(selected_indices) < same_psize:
        np.random.shuffle(remaining_indices)
        additional_needed = same_psize - len(selected_indices)
        selected_indices.extend(remaining_indices[:additional_needed])
    elif len(selected_indices) > same_psize:
        # Trim excess samples if necessary
        selected_indices = selected_indices[:same_psize]

    # Return sampled items
    return [_list[i] for i in selected_indices]

def get_smae_psize(patch,same_psize,_type='zero',min_seq_len=128,pos=None):
    ps = int(patch.size(0))

    if same_psize > 0 and same_psize < 1:
        same_psize = int(same_psize * ps)
        same_psize = max(min_seq_len,same_psize)

    same_psize = int(same_psize)

    if ps < same_psize:
        if _type == 'zero':
            patch = torch.cat([patch,torch.zeros((int(same_psize-ps),patch.size(1)))],dim=0)
            if pos is not None:
                pos = F.pad(pos, (0, 0, 0, int(same_psize-ps)), mode='constant', value=-1)
        elif _type == 'random':
            ori_indices = torch.arange(ps)
            selected_indices = torch.cat([ori_indices, torch.randint(0, ps, (int(same_psize - ps),))])
            patch = patch[selected_indices]
            if pos is not None:
                pos = pos[selected_indices]
        elif _type == 'none':
            pass
        else:
            raise NotImplementedError
    elif ps > same_psize:
        idx = torch.randperm(ps)
        patch = patch[idx.long()]
        patch = patch[:int(same_psize)]
        if pos is not None:
            indices = idx[:same_psize]
            pos = [pos[i] for i in indices]

    if pos is not None:
        return patch,pos
    
    return patch

def parse_dataframe(args,dataset):
    train_df = dataset['train']
    test_df = dataset['test']
    val_df = dataset['val']
    return train_df['ID'].tolist(),train_df['Label'].tolist(),test_df['ID'].tolist(),test_df['Label'].tolist(),val_df['ID'].tolist(),val_df['Label'].tolist()

def get_split_dfs(args,df):
    if 'Split' not in df.columns:
        raise ValueError("CSV file must contain a 'Split' column")

    train_df = df[df['Split'].str.lower() == 'train'].reset_index(drop=True)
    test_df = df[df['Split'].str.lower() == 'test'].reset_index(drop=True)
    val_df = df[df['Split'].str.lower() == 'val'].reset_index(drop=True)

    if args.val2test:
        test_df = pd.concat([val_df, test_df], axis=0).reset_index(drop=True)
        args.val_ratio = 0.

    if len(val_df) == 0:
        val_df = test_df

    return train_df, test_df, val_df

def get_data_dfs(args, csv_file):
    if args.rank == 0:
        print(f'[dataset] loading dataset from {csv_file}')

    df = pd.read_csv(csv_file)

    required_columns = ['ID', 'Split', 'Label']

    if args.datasets.lower().startswith('surv') and 'Label' not in df.columns:
        df = survival_label(df)

    if not all(col in df.columns for col in required_columns):
        if len(df.columns) == 2:
            from sklearn.model_selection import train_test_split
            df.columns = ['ID', 'Label']
            if args.rank == 0:
                print(f"[dataset] Split column not found in CSV file, splitting data randomly with val_ratio={args.val_ratio}")
            train_indices, test_indices = train_test_split(
                range(len(df)),
                test_size=args.val_ratio,  
                random_state=args.seed  
            )
            df['Split'] = 'train'
            df.loc[test_indices, 'Split'] = 'test' 
            args.val_ratio = 0.
        elif len(df.columns) == 4:
            df.columns = ['Case', 'ID', 'Label', 'Split']
        else:
            raise ValueError(f"CSV file must contain these columns: {required_columns}")

    if args.rank == 0:
        print(f"Dataset statistics:")
        print(f"Total samples: {len(df)}")
        print(f"Label distribution:")
        print(df['Label'].value_counts())
        print("Split distribution:")
        print(df['Split'].value_counts())

    return df

def get_patient_label(args,csv_file):
    df = pd.read_csv(csv_file)

    if len(df.columns) == 2:
        df.columns = ['ID', 'Label']
    else:
        df.columns = ['Case', 'ID', 'Label', 'Split']

    patients_list = df['ID']
    labels_list = df['Label']

    label_counts = labels_list.value_counts().to_dict()

    if args:
        if args.rank == 0:
            print(f"patient_len:{len(patients_list)} label_len:{len(labels_list)}")
            print(f"all_counter:{label_counts}")

    return df

def get_patient_label_surv(args,csv_file):
    if args:
        if args.rank == 0:
            print('[dataset] loading dataset from %s' % (csv_file))
    rows = pd.read_csv(csv_file)
    rows = survival_label(rows)

    label_dist = rows['Label'].value_counts().sort_index()

    if args:
        if args.rank == 0:
            print('[dataset] discrete label distribution: ')
            print(label_dist)
            print('[dataset] dataset from %s, number of cases=%d' % (csv_file, len(rows)))

    return rows

def data_split(seed,df, ratio, shuffle=True, label_balance_val=True):
    if label_balance_val:
        val_df = pd.DataFrame()
        train_df = pd.DataFrame()

        for label in df['Label'].unique():
            label_df = df[df['Label'] == label]
            n_total = len(label_df)
            offset = int(n_total * ratio)

            if shuffle:
                label_df = label_df.sample(frac=1, random_state=seed)

            val_df = pd.concat([val_df, label_df.iloc[:offset]])
            train_df = pd.concat([train_df, label_df.iloc[offset:]])
    else:
        n_total = len(df)
        offset = int(n_total * ratio)

        if n_total == 0 or offset < 1:
            return pd.DataFrame(), df

        if shuffle:
            df = df.sample(frac=1, random_state=seed)

        val_df = df.iloc[:offset]
        train_df = df.iloc[offset:]

    return val_df, train_df

def get_kfold(args,k, df, val_ratio=0, label_balance_val=True):
    if k <= 1:
        raise NotImplementedError

    skf = StratifiedKFold(n_splits=k)

    train_dfs = []
    test_dfs = []
    val_dfs = []

    for train_index, test_index in skf.split(df, df['Label']):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        if val_ratio != 0:
            val_df, train_df = data_split(args.seed,train_df, val_ratio, True, label_balance_val)

            if args.val2test:
                test_df = pd.concat([val_df, test_df], axis=0).reset_index(drop=True)
                args.val_ratio = 0.
        else:
            val_df = pd.DataFrame()

        train_dfs.append(train_df)
        test_dfs.append(test_df)
        val_dfs.append(val_df)

    return train_dfs, test_dfs, val_dfs

def survival_label(rows):
    n_bins, eps = 4, 1e-6
    uncensored_df = rows[rows['Status'] == 1]
    disc_labels, q_bins = pd.qcut(uncensored_df['Event'], q=n_bins, retbins=True, labels=False)
    q_bins[-1] = rows['Event'].max() + eps
    q_bins[0] = rows['Event'].min() - eps
    disc_labels, q_bins = pd.cut(rows['Event'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
    # missing event data
    disc_labels = disc_labels.values.astype(int)
    disc_labels[disc_labels < 0] = -1
    rows.insert(len(rows.columns), 'Label', disc_labels)
    return rows

def collate_fn_nbs(batch):
    return batch[0]

def collate_fn_img_batch_list(batch):
    first_item = batch[0]
    batch_size = len(batch)
    labels = []
    inputs = []

    for i, item in enumerate(batch):
        labels.append(item['target'])
        inputs.append(item['input'])
    outputs = {
        'input': inputs,
        'target': torch.cat(labels, dim=0)
    }

    optional_keys = ['pos', 'idx','event','censorship']
    for key in optional_keys:
        if key in first_item:
            outputs[key] = torch.cat([item[key] for item in batch if key in item], dim=0)

    return outputs

def collate_fn_img_batch(batch):

    first_item = batch[0]
    batch_size = len(batch)

    _images = torch.empty((first_item['input'].size(0) * batch_size, *first_item['input'].shape[1:]), 
                         dtype=first_item['input'].dtype, 
                         device=first_item['input'].device,
                         memory_format=torch.channels_last)

    labels = []
    
    for i, item in enumerate(batch):
        _images[i*first_item['input'].size(0):(i+1)*first_item['input'].size(0)].copy_(item['input'])
        labels.append(item['target'])

    outputs = {
        'input': _images,
        'target': torch.cat(labels, dim=0)
    }

    optional_keys = ['pos', 'idx','event','censorship']
    for key in optional_keys:
        if key in first_item:
            outputs[key] = torch.cat([item[key] for item in batch if key in item], dim=0)

    return outputs

class MultiScaleDataset(Dataset):
    def __init__(self, dataset1, dataset2, image_input=True,shuffle=False):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.image_input = image_input
        self.shuffle = shuffle
        assert len(dataset1) == len(dataset2), "Datasets must have equal length"

    def __len__(self):
        return len(self.dataset1)

    def set_epoch(self, epoch):
        self.dataset1.set_epoch(epoch)
        self.dataset2.set_epoch(epoch)

    def __getitem__(self, idx):
        data1 = self.dataset1[idx]
        data2 = self.dataset2[idx]

        output = {}

        input1, input2 = data1['input'], data2['input']
        
        if self.image_input:
            batch_size1, c1, h1, w1 = input1.shape
            batch_size2, c2, h2, w2 = input2.shape
            
            is_channels_last = input1.is_contiguous(memory_format=torch.channels_last)
            
            combined_input = torch.empty(
                (batch_size1 + batch_size2, c1, h1, w1), 
                dtype=input1.dtype, 
                device=input1.device,
                memory_format=torch.channels_last if is_channels_last else torch.contiguous_format
            )
            combined_input[:batch_size1] = input1
            combined_input[batch_size1:] = input2

        else:
            combined_input = torch.cat([input1, input2], dim=0)
        
        if self.shuffle:
            indices = torch.randperm(combined_input.size(0))
            combined_input = combined_input[indices]

        output['input'] = combined_input
        output['target'] = data1['target'] 
        
        if 'feat' in data1:
            output['feat'] = torch.cat([data1['feat'], data2['feat']], dim=0)

        if 'pos' in data1:
            pos1, pos2 = data1['pos'], data2['pos']
            pos1, pos2 = data1['pos'], data2['pos']
    
            # Extract pos_all (first row) from each tensor
            pos1_all = pos1[0]  # [2]
            pos2_all = pos2[0]  # [2]
            
            # Remove pos_all and concatenate remaining positions
            pos_combined = torch.cat([pos1[1:], pos2[1:]], dim=0)  # [2N, 2]
            
            # Create expanded pos_all for each point
            pos_all = torch.cat([
                pos1_all.expand(len(pos1)-1, -1),  # Expand for first N points
                pos2_all.expand(len(pos2)-1, -1)   # Expand for second N points
            ], dim=0)  # [2N, 2]
            
            # Concatenate along last dimension
            output['pos'] = torch.cat([pos_all, pos_combined], dim=1)  # [2N, 4]
            
        if 'idx' in data1:
            output['idx'] = data1['idx'] 

        if 'censorship' in data1:
            output['censorship'] = data1['censorship'] 
        
        if 'event' in data1:
            output['event'] = data1['event']

        return output


