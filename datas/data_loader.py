import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib as plt
from PIL import Image
import os
from skimage import morphology,draw
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries 

def calSK(image):
    image = np.array(image)
    image[image != 0] = 1
    skeleton =morphology.skeletonize(image)
    r = np.zeros( (image.shape[0], image.shape[1]), dtype=np.float32)
    r[skeleton == 1] = 1
    return r


class ImageFolder(data.Dataset):
	def __init__(self, root, args, image_size=224,mode='train',augmentation_prob=0.4):
		"""Initializes image paths and preprocessing module."""
		self.root = root
		
		# GT : Ground Truth
		#self.GT_paths = root[:-1]+'_GT/'
		if args.is_continuely and mode=='valid':
			self.type = 'multitask'
		else:
			self.type = args.type
		self.image_path = root+'/imgs/'
		self.is_demo = args.is_demo
		if not args.is_demo:
			if args.type == 'blood' or args.type == 'multitask':
				self.GT_paths = root+'/bloods/'
			if args.type == 'choroid':
				self.GT_paths = root+'/choroids/'
			self.GT_choroid_paths = root+'/choroids/'
		#self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
		tmp_dir = os.listdir(self.image_path)
		self.image_paths = []
		for i in tmp_dir:
			if i == '.DS_Store':
				continue
			#print(i)
			self.image_paths.append(i)
		self.image_size = image_size
		self.mode = mode
		self.RotationDegree = [-45, 0, 45]
		self.augmentation_prob = augmentation_prob
		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""

		image_path = self.image_path + self.image_paths[index]
		filename = self.image_paths[index][:-len(".jpg")]

		if not self.is_demo:
			GT_path = self.GT_paths + filename + '.npy'
			if self.type == 'choroid' or self.type == 'multitask':
				GT_choroid_path = self.GT_choroid_paths + filename + '.npy'

		image = Image.open(image_path)

		if not self.is_demo:
			GT = np.uint8(np.load(GT_path))
			if self.type == 'choroid':
				GT[GT == 1] = 0
				GT[GT == 2] = 1
			if self.type == 'multitask':
				GT_choroid = np.uint8(np.load(GT_choroid_path))
				GT_choroid[GT_choroid == 1] = 0
				GT_choroid[GT_choroid == 2] = 1

		o_h = image.size[0]
		o_w = image.size[1]
		aspect_ratio = image.size[0]/image.size[1]

		Transform =[]

		if (self.mode == 'train') and random.random() <= self.augmentation_prob:

			
			RotationDegree = random.randint(0,2)
			RotationDegree = self.RotationDegree[RotationDegree]
			#if (RotationDegree == 90) or (RotationDegree == 270):
			#	aspect_ratio = 1/aspect_ratio

			Transform.append(T.RandomRotation((RotationDegree,RotationDegree)))
						
			RotationRange = random.randint(-20,20)
			Transform.append(T.RandomRotation((RotationRange,RotationRange)))
			
			CropRange = random.randint(400,500)
			Transform.append(T.CenterCrop((int(CropRange*aspect_ratio),CropRange)))
			

			Transform = T.Compose(Transform)
			image = Transform(image)
			GT = Image.fromarray(np.uint8(GT))
			GT = Transform(GT)
			
			if self.type == 'multitask':
				GT_choroid = Image.fromarray(np.uint8(GT_choroid))
				GT_choroid = Transform(GT_choroid)
			
			
			ShiftRange_left = random.randint(0,20)
			ShiftRange_upper = random.randint(0,20)
			ShiftRange_right = image.size[0] - random.randint(0,20)
			ShiftRange_lower = image.size[1] - random.randint(0,20)
			image = image.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))
			GT = GT.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))

			if self.type == 'multitask':
				GT_choroid = GT_choroid.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))
			
			if random.random() < 0.5:
				image = F.hflip(image)
				GT = F.hflip(GT)
				if self.type == 'multitask':
					GT_choroid = F.hflip(GT_choroid)

			h = image.size[0]
			w = image.size[1]
			#print(h, w, o_h, o_w, aspect_ratio, int(CropRange*aspect_ratio),CropRange)


			Transform =[]
			#print(image.shape, o_w, o_h)
			Transform.append(T.Resize([o_w, o_h], interpolation=Image.BILINEAR))
			Transform = T.Compose(Transform)
			image = Transform(image)



			Transform =[]
			Transform.append(T.Resize([o_w, o_h], interpolation=Image.NEAREST))
			Transform = T.Compose(Transform)
			GT = Transform(GT)
			if self.type == 'multitask':
				GT_choroid = Transform(GT_choroid)


			#plt.imsave("blood.jpg", mark_boundaries(np.array(image), np.array(GT)) )
			#plt.imsave("choroid.jpg", mark_boundaries(np.array(image), np.array(GT_choroid)) )

			Transform =[]
			Transform.append(T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02))
			Transform.append(T.ToTensor())
			Transform = T.Compose(Transform)
			image = Transform(image)
		else:
			Transform =[]
			Transform.append(T.ToTensor())
			Transform = T.Compose(Transform)
			image = Transform(image)

		if not self.is_demo:
			if self.type == 'multitask':
				GT_choroid = np.array(GT_choroid)
				GT_choroid = torch.from_numpy(GT_choroid)
				#GT_choroid[GT_choroid < 0.5] = 0
				#GT_choroid[GT_choroid != 0] = 1
			GT  = np.array(GT)
			GT = torch.from_numpy(GT)
			#GT[GT < 0.5] = 0
			#GT[GT != 0] = 1

		#####################
		Transform =[]
		Transform.append(T.ToTensor())
		Transform = T.Compose(Transform)

		Norm_ = T.Normalize((0.5), (0.5))
		image = Norm_(image)[0, :, :]
		image = image.unsqueeze(0)

		if self.is_demo:
			GT = torch.zeros(image.size())
			if self.type == 'multitask':
				GT_choroid = torch.zeros(image.size())

		if self.type != 'multitask':
			return image, GT, filename
		else:
			return image, [GT, GT_choroid], filename

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)

def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def get_loader(image_path, image_size, batch_size, args, num_workers=2, mode='train',augmentation_prob=0.4):
	"""Builds and returns Dataloader."""

	GLOBAL_WORKER_ID = None
	
	dataset = ImageFolder(root = image_path, args=args, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)

	if mode == 'train':
		data_loader = data.DataLoader(dataset=dataset,
									  batch_size=batch_size,
									  shuffle=True,
									  num_workers=num_workers)
	else:
		data_loader = data.DataLoader(dataset=dataset,
									  batch_size=batch_size,
									  shuffle=False,
									  num_workers=num_workers)
	return data_loader
