from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Example: load a dummy PyTorch model (replace with real model later)
class DummyModel:
    def __call__(self, x):
        # Simulate prediction: return random
        import random
        return torch.tensor([[random.random(), random.random()]])

model = DummyModel()

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    # Read image
    img = Image.open(file.file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    # Make prediction (replace with real model inference)
    outputs = model(img_tensor)
    pred_idx = torch.argmax(outputs, dim=1).item()
    result = "Cancer Detected" if pred_idx == 0 else "No Cancer Detected"
    confidence = float(outputs[0][pred_idx]) * 100

    return {"result": result, "confidence": round(confidence, 2)}
