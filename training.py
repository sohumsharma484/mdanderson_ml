import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from models.unet.model import *
from models.resunet.model import *
from SegModel import SegModel
from utils import *
from custom_datasets import *
import albumentations as A

# Only need to use this file to train on Windows. pl_training notebook can be used for linux

# Github from https://github.com/akshaykulkarni07/pl-sem-seg
BATCH_SZ = 4
LEARNING_RATE = 1e-3
EPOCHS = 1
train_indices = [i for i in range(3)]
val_indices = [i + 3 for i in range(1)]
test_indices = [i + 1064 for i in range(95)]
channels = 3
num_classes = 10
filenames = "index_to_filenames.pkl"
class_weights = None


# raw_model inherits nn.module. This is what to change for a different model
raw_model = UNet(num_classes = num_classes, num_channels=channels, bilinear = False)
# raw_model = build_resunetplusplus(num_classes=num_classes, num_channels=channels)

# Build transform based on number of channels
if channels == 3:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.35675976, 0.37380189, 0.3764753], std = [0.32064945, 0.32098866, 0.32325324])
    ])
elif channels == 2:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.35675976, 0.37380189], std = [0.32064945, 0.32098866])
    ])
elif channels == 1:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.35675976], std = [0.3206494])
    ])
else:
    raise Exception("Not valid number of channels")

augmentation = A.Compose(
    [
        A.Resize(width=256, height=256),
        # A.RandomCrop(width=128, height=128),
        A.Rotate(limit=40, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.05),
        # A.Blur(blur_limit=3, p=0.5),
    ]
)

trainset = tissue_dataset(train_indices, filenames, transform=transform, augment=augmentation)
valset = tissue_dataset(val_indices, filenames, transform=transform, augment=augmentation)
testset = tissue_dataset(test_indices, filenames, transform=transform, augment=None)
train_dataloader = DataLoader(trainset, batch_size = BATCH_SZ, shuffle = True, pin_memory=True)
val_dataloader = DataLoader(valset, batch_size = 1, shuffle = False, pin_memory=True)

# Below model encapsulates above model in a lightning object for training
model = SegModel(raw_model, train_dataloader, val_dataloader, class_weights=class_weights, channels=channels, batch_size=BATCH_SZ, learning_rate=LEARNING_RATE)

# Saves model after best epoch (smallest loss)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath = 'checkpoints_unet/',
    verbose = True, 
    monitor = 'validation_loss',
    mode = 'min',
)



trainer = pl.Trainer(max_epochs=EPOCHS, callbacks = [checkpoint_callback], log_every_n_steps=1)
trainer.fit(model)