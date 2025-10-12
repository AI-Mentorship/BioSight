import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import os
import splitfolders

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm


class LungDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes


# split using splitfolders seed 1234


labels = ['aca', 'benign', 'scc']
transform = transforms.Compose([transforms.ToTensor()])
dataset = LungDataset(data_dir='./output/train')
target_to_class = {v: k for k, v in ImageFolder('./output/train').class_to_idx.items()}
print(target_to_class)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for image, label in dataset:
    break

for images, labels in dataloader:
    break

labels.shape