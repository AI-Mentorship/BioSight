# backend/models/oral_cancer.py
import torch
import torch.nn as nn
import timm

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
        return self.classifier(x)
