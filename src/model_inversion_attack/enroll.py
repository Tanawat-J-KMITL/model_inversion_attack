import os
import random
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from model_inversion_attack.model import BankingFaceModel


def run_enroll(args):

    if not args.input:
        args.input = "./secrets/img"

    if not args.output:
        args.output = "./secrets/features"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_model = BankingFaceModel(device)

    # FaceNet expects 160x160 and normalized to [-1, 1]
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # [-1,1]
    ])

    IMG_DIR = Path(args.input)
    FEATURE_DIR = Path(args.output)

    IMG_DIR.mkdir(exist_ok=True)
    FEATURE_DIR.mkdir(exist_ok=True)

    print(f"Drop enrollment images into {args.input}")
    input("Press ENTER when ready...")

    # Collect images
    image_paths = [
        IMG_DIR / f
        for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(image_paths) < 2:
        raise RuntimeError("Need at least 2 images for enrollment.")

    random.shuffle(image_paths)

    split_idx = int(0.8 * len(image_paths))
    train_paths = image_paths[:split_idx]
    test_paths = image_paths[split_idx:]

    def load_embedding(path):
        img = Image.open(path).convert("RGB")
        img = preprocess(img).unsqueeze(0).to(device)
        return face_model.embed(img)

    # Build template from training set
    train_embeddings = [load_embedding(p) for p in train_paths]

    template = torch.cat(train_embeddings, dim=0).mean(dim=0, keepdim=True)
    template = torch.nn.functional.normalize(template, p=2, dim=1)

    # Validate on test set
    confidences = []
    for p in test_paths:
        emb = load_embedding(p)
        conf = face_model.compute_confidence(emb, template)
        confidences.append(conf)
        print(f"{p.name} → confidence: {conf:.4f}")

    mean_conf = sum(confidences) / len(confidences)
    print(f"\nMean test confidence: {mean_conf:.4f}")

    THRESHOLD = 0.82  # realistic banking-style threshold

    if mean_conf >= THRESHOLD:
        name = input("User ID to store template as: ").strip()
        if name:
            torch.save(template.cpu(), FEATURE_DIR / f"{name}.pt")
            print("Features extracted successfully ✅")
    else:
        print("Enrollment failed ❌ (insufficient similarity)")
