import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import kagglehub
from PIL import Image
import os
import shutil
from tqdm.notebook import tqdm


dataset_path = "/content/kagglehub_data/obulisainaren/multi-cancer"


if os.path.exists(dataset_path):
    shutil.rmtree(dataset_path)

path = kagglehub.dataset_download("obulisainaren/multi-cancer")

print("Path to dataset files:", path)

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])

oral_cancer_dir = os.path.join(path, "Multi Cancer", "Multi Cancer", "Oral Cancer")

oral_dataset = datasets.ImageFolder(root=oral_cancer_dir, transform=transform)
print(oral_dataset.class_to_idx)


oral_dataset = datasets.ImageFolder(root=oral_cancer_dir, transform=transform)


oral_loader = DataLoader(oral_dataset, batch_size=32, shuffle=True)

print("Classes:", oral_dataset.classes)
print("Number of images:", len(oral_dataset))
print("Class to index mapping:", oral_dataset.class_to_idx)

total_size = len(oral_dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

train_dataset, test_dataset = random_split(
    oral_dataset, [train_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class OralCancerClassifier(nn.Module):
  def __init__(self, num_classes=2):
      super(OralCancerClassifier, self).__init__()
      self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
      self.features = nn.Sequential(*list(self.base_model.children())[:-1])
      enet_out_size = 1280
      self.classifier = nn.Linear(enet_out_size, num_classes)

  def forward(self, x):
      x = self.features(x)
      x = torch.flatten(x, 1)
      output = self.classifier(x)
      return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OralCancerClassifier(num_classes=2).to(device)

print("Using device:", device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader):.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {100*correct/total:.2f}%")
