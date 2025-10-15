import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm

print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__)
print('Pandas version', pd.__version__)

class ColonCancerDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes


transform = transforms.Compose([transforms.ToTensor()])
dataset = ColonCancerDataset(data_dir='./colon_output/train', transform=transform)
len(dataset)
target_to_class = {v: k for k, v in ImageFolder('./colon_output/train').class_to_idx.items()}
print(target_to_class)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
for image, label in dataset:
    break

for images, labels in dataloader:
    break

class ColonCancerClassifer(nn.Module):
    def __init__(self, num_classes=3):
        super(ColonCancerClassifer, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output


        