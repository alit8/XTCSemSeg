import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import glob
import random
from skimage import io

IN_DOMAINS = ['rgb']
OUT_DOMAINS = ['seg']

class DD2Dataset(Dataset):

    def __init__(self, root_dir, mode, split=0, augment=True, input_size=(256, 256), in_domain='rgb', out_domain='seg'):
        
        self.root_dir = root_dir
        self.mode = mode
        self.augment = augment
        self.input_size = input_size
        self.scale_size = tuple([i + 32 for i in input_size])
        self.in_domain = in_domain
        self.out_domain = out_domain
        self.mean = [107.73358698, 104.88655131, 101.54991014]
        self.std = [54.11557331, 55.60648951, 58.84428772]

        input_dir = os.path.join(self.root_dir, self.in_domain)
        
        total = len(glob.glob1(input_dir, "*.png"))
        
        indices = list(range(total))
        split = int(np.floor(split * total))

        if self.mode == 'train':
          self.indices = indices[split:]
        elif self.mode == 'val':
          self.indices = indices[:split]

        assert (self.in_domain in IN_DOMAINS), 'Not a valid input domain'
        assert (self.out_domain in OUT_DOMAINS), 'Not a valid output domain'

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = self.indices[idx]

        img_name = "Door" + "%04d.png" % (idx + 1)
        
        in_path = os.path.join(self.root_dir, self.in_domain, img_name)
        out_path = os.path.join(self.root_dir, self.out_domain, img_name)

        PILImage = transforms.ToPILImage()
        colorJitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        resize_scale = transforms.Resize(self.scale_size, interpolation=2)
        resize_scale_nearest = transforms.Resize(self.scale_size, interpolation=0)
        hFlip = random.random() > 0.5
        tensor = transforms.ToTensor()
        norm = transforms.Normalize(self.mean, self.std)
        resize = transforms.Resize(self.input_size, interpolation=2)
        resize_nearest = transforms.Resize(self.input_size, interpolation=0)

        if self.in_domain == 'rgb':
          x = cv2.imread(in_path, cv2.IMREAD_COLOR)
          x = x[:, :, ::-1] # bgr -> rgb
          x = PILImage(x)

          if self.augment:
            x = colorJitter(x)
            x = resize_scale(x)
            i, j, h, w = transforms.RandomCrop.get_params(x, self.input_size)
            x = TF.crop(x, i, j, h, w)
            
            if hFlip:
              x = TF.hflip(x)
          else:
            x = resize(x)

          x = tensor(x)
          # x = norm(x)
          x = np.asarray(x, np.float32)


        if self.out_domain == 'seg':
          y = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
          y = PILImage(y)

          if self.augment:
            y = resize_scale_nearest(y)
            y = TF.crop(y, i, j, h, w)

            if hFlip:
              y = TF.hflip(y)
          else:
            y = resize_nearest(y)

          y = np.asarray(y, np.int64)

          y[y == 2] = 1

        edge_map = generate_edge_map_from(y)

        item = {
            'x': x,
            'y': y,
            'edge': edge_map
        }

        return item

    
def generate_edge_map_from(label):
  lbl = cv2.GaussianBlur(label.astype('uint8'), (3, 3), 0)
  edge = cv2.Laplacian(lbl, cv2.CV_64F)
  activation = cv2.dilate(np.abs(edge), np.ones((5, 5), np.uint8), iterations=1)
  activation[activation != 0] = 1
  return cv2.GaussianBlur(activation, (15, 15), 5)

