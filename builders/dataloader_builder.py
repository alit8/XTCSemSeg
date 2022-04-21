import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

from datasets.nyuv2 import NYUv2Dataset
from datasets.ade20k import ADE20KDataset
from datasets.deepdoors2 import DD2Dataset
from datasets.doort import DoortDataset

def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def build_dataloaders(dataset,
                      in_domain,
                      out_domain,
                      input_size,
                      batch_size,
                      augment,
                      valid_size=0.2,
                      shuffle=True,
                      show_sample=False,
                      num_workers=4,
                      pin_memory=False):

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    if dataset in ['ade20k', 'ade_nyu']:
      
      ade20k_train = ADE20KDataset(
          root_dir="../ADE20K", mode='train', split=valid_size, augment=augment, input_size=input_size, in_domain=in_domain, out_domain=out_domain
      )

      ade20k_val = ADE20KDataset(
          root_dir="../ADE20K", mode='val', split=valid_size, augment=None, input_size=input_size, in_domain=in_domain, out_domain=out_domain
      )

    if dataset in ['nyuv2', 'ade_nyu']:

      nyuv2_train = NYUv2Dataset(
          root_dir="../NYUv2", mode='train', split=valid_size, augment=augment, input_size=input_size, in_domain=in_domain, out_domain=out_domain
      )

      nyuv2_val = NYUv2Dataset(
          root_dir="../NYUv2", mode='val', split=valid_size, augment=None, input_size=input_size, in_domain=in_domain, out_domain=out_domain
      )

    if dataset in ['dd2', 'door']:

      dd2_train = DD2Dataset(
          root_dir="../DeepDoors2", mode='train', split=valid_size, augment=augment, input_size=input_size, in_domain=in_domain, out_domain=out_domain
      )

      dd2_val = DD2Dataset(
          root_dir="../DeepDoors2", mode='val', split=valid_size, augment=None, input_size=input_size, in_domain=in_domain, out_domain=out_domain
      )

    if dataset in ['doort', 'door']:

      doort_train = DoortDataset(
          root_dir="../Doort", mode='train', split=valid_size, augment=augment, input_size=input_size, in_domain=in_domain, out_domain=out_domain
      )

      doort_val = DoortDataset(
          root_dir="../Doort", mode='val', split=valid_size, augment=None, input_size=input_size, in_domain=in_domain, out_domain=out_domain
      )
      

    if dataset == 'ade20k':

      train = ade20k_train
      val = ade20k_val
    
    elif dataset == 'nyuv2':

      train = nyuv2_train
      val = nyuv2_val

    elif dataset == 'ade_nyu':

      train = ConcatDataset([ade20k_train, nyuv2_train])
      val = ConcatDataset([ade20k_val, nyuv2_val])

    elif dataset == 'dd2':
      train = dd2_train
      # train = ConcatDataset([ade20k_train, nyuv2_train, dd2_train])
      # val = ConcatDataset([ade20k_val, nyuv2_val])

      val = dd2_val
    
    elif dataset == 'door':

      train = ConcatDataset([dd2_train, doort_train])
      val = ConcatDataset([dd2_val, doort_val])

    elif dataset == 'doort':

      train = doort_train
      val = doort_val

    
    num_train = len(train)
    num_val = len(val)
    # indices = list(range(num_train))
    # split = int(np.floor(valid_size * num_train))

    print("Train dataset size: {}".format(num_train))
    print("Validation dataset size: {}".format(num_val))

    # if shuffle:
    #     np.random.shuffle(indices)

    # train_idx, valid_idx = indices[split:], indices[:split]
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    if show_sample:
      
      print("10 random samples:")

      cmap = colormap()
      
      fig, axs = plt.subplots(10, 2)

      for i, ind in enumerate(range(10)):
        rgb_image, seg_image = train[ind]

        axs[i, 0].imshow(rgb_image)
        axs[i, 0].axis('off')

        color_seg_image = cmap[seg_image.astype(np.uint8)]

        axs[i, 1].imshow(color_seg_image)
        axs[i, 1].axis('off')

      plt.show()

    train_loader = DataLoader(
        train, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory, shuffle=True
    )
    valid_loader = DataLoader(
        val, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)











