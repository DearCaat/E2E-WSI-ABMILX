import time
import os
import argparse
import pdb
import pickle
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np

import lmdb
import io

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2

class Whole_Slide_Bag_FPa(Dataset):
	def __init__(self,
		file_path,
		wsi,
		save_img,
		img_path,
		img_transforms=None,
		img_size=0):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			wsi (openslide): Openslide object representing the whole slide image.
			save_img (bool): Whether to save images to disk.
			img_path (string): Path to save images.
			img_transforms (callable, optional): Optional transform to be applied on a sample.
			img_size (int): Size to resize images to (0 means no resizing).
		"""
		self.wsi = wsi
		if img_size == 0:
			self.roi_transforms = v2.ToImage()
		else:
			self.roi_transforms = v2.Compose([
				v2.ToImage(),  # Convert to tensor
				v2.Resize(size=(img_size, img_size), antialias=True),  # Resize with antialiasing
			])
		self.save_img = save_img
		self.img_path = img_path
		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		"""Print summary information about the dataset"""
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		"""
		Get a patch from the WSI at the given index
		
		Args:
			idx (int): Index of the patch to retrieve
			
		Returns:
			dict: Dictionary containing the image, coordinates, and index
		"""
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		
		if self.save_img and not os.path.isfile(os.path.join(self.img_path,str(idx)+'.jpg')):
			img.save(os.path.join(self.img_path,str(idx)+'.jpg'),quality=args.img_quality)

		if self.roi_transforms:
			img = self.roi_transforms(img)
		return {'img': img, 'coord': coord, 'idx': idx}

def tensor_to_pil(image):
	"""Convert tensor to PIL Image"""
	unloader = v2.ToPILImage()
	image = image.squeeze(0)  # Remove batch dimension
	image = unloader(image)
	return image

def compress_image(image, quality=80):
	"""Compress image to JPEG format with given quality"""
	img_buffer = io.BytesIO()
	image.save(img_buffer, format='JPEG', quality=quality)
	return img_buffer.getvalue()

def dump_pickle(obj):
	"""Serialize object using pickle"""
	return pickle.dumps(obj)

def load_pickle(buf):
	"""Deserialize object using pickle"""
	return pickle.loads(buf)

def get_processed_slides(progress_file):
	"""Get list of already processed slides from progress file"""
	if os.path.exists(progress_file):
		with open(progress_file, 'r') as f:
			return set(f.read().splitlines())
	return set()

def compute_w_loader(args, txn, global_count, loader, save_lmdb, verbose=0, slide_name=None, db=None):
	"""Process patches using data loader and save to LMDB if requested"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches')

	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():    
			batch = data['img']
			global_count += len(batch)
			idx = data['idx']
		if save_lmdb:
			for i, _img in enumerate(batch):
				_img = tensor_to_pil(_img)
				txn.put(u'{}-{}'.format(slide_name, idx[i]).encode('ascii'), dump_pickle(compress_image(_img,quality=args.img_quality)))
				if (count * args.batch_size + i) % 5000 == 0:
					txn.commit()
					txn = db.begin(write=True)
	if save_lmdb:    
		txn.commit()
		txn = db.begin(write=True)
		
	return global_count, txn

def process_slides(args, bags_dataset, total):
	"""Process all slides in the dataset"""
	progress_file = os.path.join(args.patch_dir, str(args.lmdb_name)+'_progress.txt')
	processed_slides = get_processed_slides(progress_file)

	global_count = 0
	loader_kwargs = {'num_workers': args.workers}
	txn = None
	db = None
	if args.save_lmdb:
		final_db_path = os.path.join(args.patch_dir, args.lmdb_name+'.lmdb')
		map_size = 2199023255552  # ~2TB, adjust based on OS/db size requirements
		db = lmdb.open(final_db_path, subdir=False,
					   map_size=map_size, meminit=False, map_async=True)
		txn = db.begin(write=True)

	for bag_candidate_idx in tqdm(range(total)):
		try:
			slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]

			if slide_id in processed_slides:
				if args.rank == 0:
					print(f'Skipping already processed slide: {slide_id}')
				continue

			bag_name = slide_id + '.h5'
			h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
			slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)

			if args.rank == 0:
				print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
				print(slide_id)

			img_path = os.path.join(args.patch_dir, 'imgs', slide_id)
			lmdb_path = os.path.join(args.patch_dir, 'lmdb', slide_id)
			if args.save_img:
				os.makedirs(img_path, exist_ok=True)
			if args.save_lmdb:
				os.makedirs(lmdb_path, exist_ok=True)

			with h5py.File(h5_file_path, "r") as f:
				dset = f['coords']
				len_img = len(dset)

			if args.save_img:
				if not args.no_auto_skip and len(os.listdir(img_path)) == len_img:
					if args.rank == 0:
						print('skipped {}'.format(slide_id))
					continue 

			if args.save_lmdb:
				if not args.no_auto_skip and os.path.isfile(os.path.join(lmdb_path, 'data.lmdb')):
					continue

			time_start = time.time()
			wsi = openslide.open_slide(slide_file_path)
			dataset = Whole_Slide_Bag_FPa(file_path=h5_file_path, 
										  wsi=wsi, save_img=args.save_img, img_path=img_path, img_size=args.img_size)

			if args.distributed:
				loader = DataLoader(dataset=dataset, shuffle=False, batch_size=args.batch_size, 
									sampler=DistributedSampler(dataset), **loader_kwargs)
			else:
				loader = DataLoader(dataset=dataset, shuffle=False, batch_size=args.batch_size, **loader_kwargs)

			global_count, txn = compute_w_loader(args, txn, global_count, loader=loader, 
											save_lmdb=args.save_lmdb, verbose=1, slide_name=slide_id, db=db)
			time_elapsed = time.time() - time_start
			if args.rank == 0:
				print('\ncomputing features took {} s'.format(time_elapsed))

			# Update progress
			processed_slides.add(slide_id)
			with open(progress_file, 'a') as f:
				f.write(slide_id + '\n')

		except Exception as e:
			print(f"Error occurred while processing slide {slide_id}: {e}")
			raise

	if args.save_lmdb:
		keys = []
		slide_ids = []
		PN_dict = {}
		for bag_candidate_idx in range(total):
			slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
			slide_ids.append(slide_id)
			bag_name = slide_id + '.h5'
			h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
			with h5py.File(h5_file_path, "r") as f:
				dset = f['coords']
				len_img = len(dset)
			PN_dict[str(slide_id)] = len_img
			for k in range(len_img):
				keys.append(u'{}-{}'.format(slide_id, k).encode('ascii'))

		txn.commit()
		with db.begin(write=True) as txn:
			txn.put(b'__keys__', dump_pickle(keys))
			txn.put(b'__len__', dump_pickle(len(keys)))
			txn.put(b'__slide__',dump_pickle(slide_ids))
			txn.put(b'__pn__', dump_pickle(PN_dict))

		db.sync()
		db.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Feature Extraction')
	parser.add_argument('--data_h5_dir', type=str, default=None)
	parser.add_argument('--data_slide_dir', type=str, default=None)
	parser.add_argument('--slide_ext', type=str, default= '.svs')
	parser.add_argument('--lmdb_name', type=str, default= 'data')
	parser.add_argument('--csv_path', type=str, default=None)
	parser.add_argument('--patch_dir', type=str, default=None)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--img_size', type=int, default=0)
	parser.add_argument('--workers', type=int, default=2)
	parser.add_argument('--no_auto_skip', default=False, action='store_true')
	parser.add_argument('--save_img', action='store_true')
	parser.add_argument('--img_quality', type=int, default=80)
	parser.add_argument('--save_lmdb', action='store_true')
	parser.add_argument('--target_patch_size', type=int, default=224)
	args = parser.parse_args()

	args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
	device = timm.utils.init_distributed_device(args)

	if args.rank == 0:
		print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError("CSV path must be specified")

	bags_dataset = Dataset_All_Bags(csv_path, os.path.join(args.data_h5_dir, 'patches'), args)
	os.makedirs(args.patch_dir, exist_ok=True)

	total = len(bags_dataset)

	process_slides(args, bags_dataset, total)