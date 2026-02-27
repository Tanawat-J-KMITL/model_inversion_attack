import os
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from model_inversion_attack import model

THRESHOLD = 0.82


def auth(img, user_id):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_model = model.BankingFaceModel(device)

    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FEATURE_DIR = os.path.join(BASE_DIR, "..", "..", "secrets", "features")

    user_id = user_id.strip()
    template_path = Path(f"{FEATURE_DIR}/{user_id}.pt")

    if not template_path.exists():
        print(f"Unknown user features {template_path.resolve()}")
        return (False, 0.0)

    template = torch.load(template_path).to(device)

    # image_path = input("Path to authentication image: ").strip()
    # img = Image.open(image_path).convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device)

    embedding = face_model.embed(img)
    confidence = face_model.compute_confidence(embedding, template)

    print(f"Authentication confidence: {confidence:.4f}")

    if confidence >= THRESHOLD:
        print("Access GRANTED ✅")
        return (True, confidence)

    print("Access DENIED ❌")
    return (False, confidence)
