from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from uuid import uuid4
import io
import os
import torch
import torch.nn as nn
import torchvision.models as models
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2

from .model_def import LungCancerClassifer
from .model_def import OralCancerClassifier
from .model_def import ColonCancerClassifier

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=['http://localhost:3000'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def root():
    return {"Hello": "World"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
last_layer = None

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

########################### LUNGS ##############################

LUNG_CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "lung_model.pth")
LUNG_CLASS_NAMES = ["Adenocarcinoma", "Benign", "Small Cell Cancer"] 
lung_model = LungCancerClassifer(num_classes=len(LUNG_CLASS_NAMES))
lung_checkpoint = torch.load(LUNG_CHECKPOINT_PATH, map_location=DEVICE)
lung_model.load_state_dict(lung_checkpoint["model_state"])
lung_model.to(DEVICE)
lung_model.eval()

########################### CERVIX ##############################

CERVICAL_CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "cervical_model.pth")
CERVICAL_CLASS_NAMES = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"] 
cervical_model = models.resnet50(num_classes = 5)
cervical_model.fc = nn.Linear(2048, 5)
cervical_checkpoint = torch.load(CERVICAL_CHECKPOINT_PATH, map_location=DEVICE)
cervical_model.load_state_dict(cervical_checkpoint)
cervical_model.to(DEVICE)
cervical_model.eval()

########################### ORAL ##############################

ORAL_CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "oral_cancer_model.pth")
ORAL_CLASS_NAMES = ["Normal", "Small Cell Cancer"] 
oral_model = OralCancerClassifier(num_classes=len(ORAL_CLASS_NAMES))
oral_checkpoint = torch.load(ORAL_CHECKPOINT_PATH, map_location=DEVICE)
oral_model.load_state_dict(oral_checkpoint)
oral_model.to(DEVICE)
oral_model.eval()

########################### COLON ##############################

COLON_CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "colon_model.pth")
COLON_CLASS_NAMES = ["Adenocarcinoma", "Normal"] 
colon_model = ColonCancerClassifier(num_classes=len(COLON_CLASS_NAMES))
colon_checkpoint = torch.load(COLON_CHECKPOINT_PATH, map_location=DEVICE)
colon_model.load_state_dict(colon_checkpoint)
colon_model.to(DEVICE)
colon_model.eval()

@app.post("/predict")
async def predict(
    cancer_type: str = Form(...),
    patient_id: str = Form(...),
    notes: str = Form(""),
    file: UploadFile = File(...),
):
    try:
        # Pick model + labels based on cancer_type
        if cancer_type == "lung":
            model = lung_model
            class_names = LUNG_CLASS_NAMES
            last_layer = model.base_model.conv_head
        elif cancer_type == "cervical":
            model = cervical_model
            class_names = CERVICAL_CLASS_NAMES
            last_layer = model.layer4[-1]
        elif cancer_type == "oral":
            model = oral_model
            class_names = ORAL_CLASS_NAMES
            last_layer = model.base_model.conv_head
        elif cancer_type == "colon":
            model = colon_model
            class_names = COLON_CLASS_NAMES
            last_layer = model.base_model.conv_head
        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported cancer_type: {cancer_type}"}
            )

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        pred_idx = pred_idx.item()
        conf = conf.item()
        predicted_class = class_names[pred_idx]

        def get_heatmap(model, input_tensor, target_layer, label):
            model.eval()

            gradients = []
            activations = []

            def save_gradients(module, grad_input, grad_output):
                gradients.append(grad_output[0])

            def save_activations(module, input, output):
                activations.append(output)

            input_tensor = input_tensor.to(DEVICE)
            hook_bwd = target_layer.register_backward_hook(save_gradients)
            hook_fwd = target_layer.register_forward_hook(save_activations)
            pred = model(input_tensor)
            pred[0, label].backward()

            hook_bwd.remove()
            hook_fwd.remove()

            gradients = gradients[0]
            activations = activations[0]

            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

            for i in range(activations.shape[1]):
                activations[:, i, :, :] *= pooled_gradients[i]

            heatmap = torch.mean(activations, dim=1).squeeze()
            heatmap = torch.relu(heatmap)
            heatmap = heatmap.detach().cpu().numpy()

            return heatmap

        def normalize_heatmap(heatmap):
            heatmap = heatmap - heatmap.min()
            heatmap = heatmap / heatmap.max()
            return heatmap

        def save_heatmap(heatmap, input_tensor, save_path):
            heatmap = normalize_heatmap(heatmap)

            heatmap = cv2.resize(
                heatmap,
                (384, 384),
                interpolation=cv2.INTER_CUBIC
            )

            heatmap = cv2.applyColorMap(
                np.uint8(255 * heatmap),
                cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            img = (input_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
            img = np.clip(img, 0, 1)

            alpha = 0.4
            overlay = (alpha * heatmap / 255) + ((1 - alpha) * img)
            overlay = np.clip(overlay, 0, 1)
            
            #return overlay
            plt.figure(figsize=(6,6))
            plt.imshow(overlay)
            plt.axis("off")
            plt.colorbar()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            
            img = Image.open(buf)
            img.save(save_path)
        

        heatmap = get_heatmap(model, img_tensor, last_layer, pred_idx)
        filename = f"{uuid4().hex}.png"
        save_path = os.path.join(UPLOAD_DIR, filename)
        save_heatmap(heatmap, img_tensor, save_path)
        image_url = f"http://localhost:8000/uploads/{filename}"

        return JSONResponse({
            #"case_id": case_id,
            #"patient_id": patient_id,
            "cancer_type": cancer_type,
            "predicted_class": predicted_class,
            "class_index": pred_idx,
            "confidence": conf,
            #"notes": notes,
            "image_url": image_url,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )