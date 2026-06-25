

import torch
from torchvision.utils import save_image
from U_NET import DiffusionUNet
from Diffusion import Diffusion
from dataset import AnimalDataset
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("Results", exist_ok=True)
# Load model and dataset
print("Loading model and data...")
model = DiffusionUNet().to(device)
model.load_state_dict(torch.load("Models/best_model.pth", map_location=device))
model.eval()

diffusion = Diffusion(device=device)

print("Reconstructing image from noise...\n")

# Add high noise to real image
t_noisy = 500  




# Try to reconstruct from t=500 to t=0
print(f"\nReconstructing from t={t_noisy} to t=0...")
x= torch.randn(1, 3, 32, 32, device=device)  # Match your model's input shape

print(f"Starting from random noise with range: [{x.min():.3f}, {x.max():.3f}]\n")

for t in reversed(range(1, t_noisy + 1)):
    if t % 100 == 0:
        print(f"  Step {t}: x range [{x.min():.3f}, {x.max():.3f}]")
    
    t_tensor = torch.full((1,), t, dtype=torch.long, device=device)  # ✅ ADD THIS LINE
    
    with torch.no_grad():
        predicted_noise = model(x, t_tensor)
    # ... rest stays the same ...
    # Compute mean and variance for reverse process
    sqrt_alpha_t = torch.sqrt(diffusion.alpha[t])
    sqrt_alpha_hat_t = diffusion.sqrt_alpha_hat[t]
    sqrt_1_minus_alpha_hat_t = diffusion.sqrt_1_minus_alpha_hat[t]
    
    sqrt_alpha_t = sqrt_alpha_t.view(1, 1, 1, 1)
    sqrt_alpha_hat_t = sqrt_alpha_hat_t.view(1, 1, 1, 1)
    sqrt_1_minus_alpha_hat_t = sqrt_1_minus_alpha_hat_t.view(1, 1, 1, 1)
    
    mean = (1 / sqrt_alpha_t) * (x - ((1 - diffusion.alpha[t]) / sqrt_1_minus_alpha_hat_t) * predicted_noise)
    
    if t > 1:
        alpha_hat_t_prev = diffusion.alpha_hat[t - 1]
        posterior_variance = diffusion.beta[t] * (1 - alpha_hat_t_prev) / (1 - diffusion.alpha_hat[t])
        std = torch.sqrt(posterior_variance)
        noise = torch.randn_like(x)
        x = mean + std.view(1, 1, 1, 1) * noise
    else:
        x = mean

print(f"Final reconstruction range: [{x.min():.3f}, {x.max():.3f}]")

reconstructed = (x.clamp(-1, 1) + 1) / 2
save_image(reconstructed, "Results/reconstructed.png")

print("\nSaved:")

print("  - Results/reconstructed.png")
print("\nIf reconstruction ≈ original, the reverse process works!")