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
from tqdm import tqdm


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

transform = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor()])
dataset = LungDataset(data_dir='./output/train', transform=transform)
target_to_class = {v: k for k, v in ImageFolder('./output/train').class_to_idx.items()}
print(target_to_class)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
for image, label in dataset:
    break

for images, labels in dataloader:
    break


class LungCancerClassifer(nn.Module):
    def __init__(self, num_classes=3):
        super(LungCancerClassifer, self).__init__()
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
    
model = LungCancerClassifer(num_classes=3)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_folder = dataset
test_folder = LungDataset('./output/test')

train_loader = DataLoader(train_folder, batch_size=16, shuffle=True)
test_loader = DataLoader(test_folder, batch_size=16, shuffle=True)

num_epochs = 5
train_losses, val_losses = [], []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LungCancerClassifer(num_classes=3)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

torch.cuda.empty_cache()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc='Training loop'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}")

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.title("Loss over epochs")
plt.show()