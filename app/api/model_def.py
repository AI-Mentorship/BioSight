import torch
import torch.nn as nn
import timm
import torch.nn.functional as f

class LungCancerClassifer(nn.Module):
    def __init__(self, num_classes=3):
        super(LungCancerClassifer, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=False)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output
    

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