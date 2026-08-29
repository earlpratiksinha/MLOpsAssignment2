import time
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops_api")

app = FastAPI(title="Cats vs Dogs Classification API")

MODEL_PATH = "models/best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    if model is None:
        raise HTTPException(status_code=500, detail="Model unavailable")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        cat_prob = float(probabilities[0])
        dog_prob = float(probabilities[1])
        
    pred_label = "Cat" if cat_prob > dog_prob else "Dog"
    latency_ms = round((time.time() - start_time) * 1000, 2)
    
    logger.info(f"File: {file.filename} | Prediction: {pred_label} | Latency: {latency_ms}ms")
    
    return {
        "filename": file.filename,
        "prediction": pred_label,
        "confidence": max(cat_prob, dog_prob),
        "probabilities": {"Cat": cat_prob, "Dog": dog_prob},
        "latency_ms": latency_ms
    }