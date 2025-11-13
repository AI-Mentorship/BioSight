# backend/api.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Exact Colab class definition (DO NOT CHANGE)
# -------------------------
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

# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# -------------------------
# Load model weights
model = OralCancerClassifier(num_classes=2).to(device)
pth_path = os.path.join("models", "oral_cancer_model.pth")
if os.path.exists(pth_path):
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    print(f"[INFO] Loaded model from {pth_path}")
else:
    print(f"[WARNING] Model file not found: {pth_path}")

# -------------------------
# EXACT SAME TRANSFORM AS COLAB (No Normalization)
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # Same size as Colab
    transforms.ToTensor()           # No normalization - exactly like Colab
])

# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Convert to RGB (same as Colab's ImageFolder)
        image = Image.open(file.file).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        return {"error": f"Failed to process image: {e}"}

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()

    # EXACT CLASS MAPPING FROM COLAB
    classes = ["oral_normal", "oral_scc"]  # Index 0: normal, Index 1: cancer
    class_descriptions = ["No Cancer (Normal)", "Cancer Detected (SCC)"]
    
    prediction = classes[predicted_idx]
    description = class_descriptions[predicted_idx]
    confidence = float(probs[0][predicted_idx].item() * 100)

    return {
        "prediction": prediction,
        "description": description,
        "confidence": confidence,
        "all_probabilities": {
            "oral_normal": float(probs[0][0].item() * 100),
            "oral_scc": float(probs[0][1].item() * 100)
        }
    }
