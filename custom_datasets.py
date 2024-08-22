import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import os
import cv2
import pickle
from utils import *
   
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class tissue_dataset(data.Dataset):
    def __init__(self, indices, filenames, transform = None, augment=None):
        self.transform = transform        
        self.indices = indices
        self.transform = transform
        self.augment = augment

        with open(filenames, "rb") as f:
            self.index_to_filename = pickle.load(f)

        
    def __len__(self):
        return(len(self.indices))
    
    def __getitem__(self, idx):
        idx = self.indices[idx]

        img, mask = load_tissue_imgs(self.index_to_filename[idx])

        if self.augment:
            augmentations = self.augment(image=img, masks=[mask])
            img = np.array(augmentations["image"])
            mask = np.array(augmentations["masks"][0])
        if self.transform:
            img = self.transform(img)

        return img, mask
