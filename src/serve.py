from fastapi import FastAPI, UploadFile, File
import torch
import os
from PIL import Image
import io
from torchvision import transforms
from src.model import get_model

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

# standard cifar10 transforms for inference
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

@app.on_event("startup")
def load_model():
    global model
    model = get_model('resnet18', 10).to(device)
    
    ckpt_path = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print("loaded model checkpoint")
    else:
        print(f"warning: no checkpoint found at {ckpt_path}")
        
    model.eval()

@app.get("/health")
def health():
    if model is not None:
        return {"status": "ok"}
    return {"status": "not loaded"}, 503

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(tensor)
        probs = torch.nn.functional.softmax(preds[0], dim=0)
        
    top_prob, top_idx = torch.max(probs, 0)
    
    return {
        "class": classes[top_idx.item()],
        "probability": top_prob.item()
    }
