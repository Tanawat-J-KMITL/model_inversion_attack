import argparse
from PIL import Image
import torch
import torchvision.transforms as T
from model_inversion_attack import model


def run_reconstruct(args):

    if not args.input:
        args.input = "tensor.png"

    if not args.output:
        args.output = "reconstructed.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_model = model.BankingFaceModel(device)
    pil_img = Image.open(args.input).convert("RGB")
    tensor = T.ToTensor()(pil_img).unsqueeze(0)

    print("Loaded image stats:")
    print("  shape:", tuple(tensor.shape))
    print("  dtype:", tensor.dtype)
    print("  min:", float(tensor.min()))
    print("  max:", float(tensor.max()))

    print("Embedding the model to face's model...")
    # Resize to model size if needed
    resize = T.Resize((160, 160))
    tensor = resize(tensor)

    # Normalize like auth
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.to(device)

    embedding = face_model.embed(tensor)
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    torch.save(embedding.cpu(), args.output)

    print(f"[OK] Saved tensor to {args.output}")
