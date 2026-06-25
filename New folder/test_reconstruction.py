"""
Test reconstruction: Add noise to real image, then denoise it.
If reconstruction works, the reverse process is correct.
"""

import torch
from torchvision.utils import save_image
from U_NET import DiffusionUNet
from Diffusion import Diffusion
from dataset import AnimalDataset
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("Results", exist_ok=True)

print("Loading model and data...")
model = DiffusionUNet().to(device)
model.load_state_dict(torch.load("Models/best_model.pth", map_location=device))
model.eval()

dataset = AnimalDataset("animal_data", max_images_per_class=4, num_classes=5)
dataloader = DataLoader(dataset, batch_size=1)
real_image = next(iter(dataloader)).to(device)

diffusion = Diffusion(device=device)

print("Reconstructing image from noise...\n")

# Add high noise to real image
t_noisy = 500  # Add significant noise
t_tensor = torch.full((1,), t_noisy, dtype=torch.long, device=device)

noisy_image, _ = diffusion.add_noise(real_image, t_tensor)

print(f"Real image range: [{real_image.min():.3f}, {real_image.max():.3f}]")
print(f"Noisy image at t={t_noisy} range: [{noisy_image.min():.3f}, {noisy_image.max():.3f}]")

# Save original and noisy
real_denorm = (real_image.clamp(-1, 1) + 1) / 2
noisy_denorm = (noisy_image.clamp(-1, 1) + 1) / 2
save_image(real_denorm, "Results/original.png")
save_image(noisy_denorm, "Results/noisy_t500.png")

# Try to reconstruct from t=500 to t=0
print(f"\nReconstructing from t={t_noisy} to t=0...")

x = noisy_image.clone()

for t in reversed(range(1, t_noisy + 1)):
    if t % 100 == 0:
        print(f"  Step {t}: x range [{x.min():.3f}, {x.max():.3f}]")
    
    t_tensor = torch.full((1,), t, dtype=torch.long, device=device)
    
    with torch.no_grad():
        predicted_noise = model(x, t_tensor)
    
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
print("  - Results/original.png")
print("  - Results/noisy_t500.png")
print("  - Results/reconstructed.png")
print("\nIf reconstruction ≈ original, the reverse process works!")