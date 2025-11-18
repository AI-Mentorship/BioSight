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

from .model_def import LungCancerClassifer
from .model_def import OralCancerClassifier

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
cervical_model = models.resnet50(pretrained=False)
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
        elif cancer_type == "cervical":
            model = cervical_model
            class_names = CERVICAL_CLASS_NAMES
        elif cancer_type == "oral":
            model = oral_model
            class_names = ORAL_CLASS_NAMES
        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported cancer_type: {cancer_type}"}
            )

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        filename = f"{uuid4().hex}.png"
        save_path = os.path.join(UPLOAD_DIR, filename)
        image.save(save_path)

        image_url = f"http://localhost:8000/uploads/{filename}"

        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        pred_idx = pred_idx.item()
        conf = conf.item()
        predicted_class = class_names[pred_idx]

        # You can generate a case_id here and send it back
        case_id = f"CASE-{patient_id}"

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