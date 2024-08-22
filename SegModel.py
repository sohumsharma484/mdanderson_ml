import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
from models.unet.model import *
from torchmetrics import Accuracy

# Pytorch lightning wrapper for any model segmentation for distributed learning
class SegModel(pl.LightningModule):
    def __init__(self, net, train_dataloader, val_dataloader, num_classes=10, class_weights=None, channels=3, batch_size=64, learning_rate=1e-3, device="cpu"):
        super(SegModel, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.net = net
        self.class_weights = class_weights
        self.dev = "cpu"
        self.num_classes = num_classes

        self.TRAIN_DATALOADER = train_dataloader
        self.VAL_DATALOADER = val_dataloader
        
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_nb) :
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        if self.class_weights:
            loss_val = F.cross_entropy(out, mask, weight=torch.FloatTensor(self.class_weights).to(self.dev))
        else:
            loss_val = F.cross_entropy(out, mask)
        self.log('training_loss', loss_val)
        return loss_val

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        print(self.class_weights)
        if self.class_weights:
            loss_val = F.cross_entropy(out, mask, weight=torch.FloatTensor(self.class_weights).to(self.dev))
        else:
            loss_val = F.cross_entropy(out, mask)
        self.log('validation_loss', loss_val)
        accuracy = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.dev)
        acc = accuracy(out, mask)
        self.log('accuracy', acc, on_epoch=True)
        return loss_val
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
        return [opt], [sch]
    
    def train_dataloader(self):
        return self.TRAIN_DATALOADER
    
    def val_dataloader(self):
        return self.VAL_DATALOADER
