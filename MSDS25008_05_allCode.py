# Dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

# animal dataset class to load images from the specified directory structure
class AnimalDataset(Dataset):

    def __init__(self, root_dir,max_images_per_class=20,num_classes=5):
    

        self.root_dir = root_dir
        self.images = []
        self.max_images_per_class = max_images_per_class
        self.num_classes = num_classes
# Define the image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
        self.valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG']
        class_count = 0
        img_count_per_class = 0

# Load images from the specified directory structure
        for label in os.listdir(root_dir):
            if class_count >= self.num_classes:
                break

            label_dir = os.path.join( root_dir, label)

            if os.path.isdir(label_dir):
                img_count_per_class = 0
# Load images for the current class
                for img_name in os.listdir(label_dir):
                    if os.path.splitext(img_name)[1].lower() not in self.valid_extensions:
                        continue
                    img_path = os.path.join(label_dir, img_name)
                    try:
                        img = Image.open(img_path) 
                        img.verify()  # Verify that the image is not corrupted
                        
                        self.images.append(img_path)
                        img_count_per_class += 1
                          # Verify that the image is not corrupted

                        if img_count_per_class >= self.max_images_per_class:
                          break
                    except Exception as e:
                     print(f"Error loading image {img_name}")
                     continue
                if img_count_per_class > 0:
                    class_count += 1
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert("RGB")
            image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image: {self.images[idx]}")
            return self.__getitem__((idx + 1) % len(self.images))
# U_NET.py
import torch
import torch.nn as nn
# U-Net architecture for the denoising network in the diffusion model.
class DoubleConv(nn.Module):
    
    # constructor for the DoubleConv class
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),  # Using GroupNorm instead of BatchNorm
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
           
        )
    def forward(self, x):
            return self.encoder(x)
# U-Net architecture for the denoising network in the diffusion model.
class DiffusionUNet(nn.Module):
           
            
            def __init__(self):
                super().__init__()
                # Encoder
                self.down1 = DoubleConv(3, 64)
                self.pool1 = nn.MaxPool2d(2)
        
                self.down2 = DoubleConv(64, 128)
                self.pool2 = nn.MaxPool2d(2)
        
           # Bottleneck
                self.middle = DoubleConv(128, 256)
        
        # Time embedding
                self.time_embedding = nn.Embedding(1000, 256)
        
        # Decoder
                self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
                self.up_conv1 = DoubleConv(256, 128)
        
                self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
                self.up_conv2 = DoubleConv(128, 64)
        
        # Output
                self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
            def forward(self,x,t):
                
                down1 = self.down1(x)
                down2 = self.down2(self.pool1(down1))
                middle = self.middle(self.pool2(down2))
                t_emb = self.time_embedding(t)
                t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # [batch, 256, 1, 1]
                middle = middle + t_emb  # Add time embedding to bottleneck
                u1 = self.up1(middle)
                u1 = torch.cat([u1, down2], dim=1)  # Skip connection
                u1 = self.up_conv1(u1)
                u2 = self.up2(u1)
                u2 = torch.cat([u2, down1], dim=1)  # Skip connection
                u2 = self.up_conv2(u2)
                return self.final_conv(u2)
# Diffusion.py


import torch
# Diffusion model for image generation using DDPM reverse process.
class Diffusion:
    
    
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.noise_steps = noise_steps
        self.device = device
        
        self.beta = torch.linspace(beta_start, beta_end, noise_steps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        # Pre-compute for efficiency
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_1_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat)
# Sample random timesteps for training.
    def sample_timesteps(self, batch_size):
        
        return torch.randint(low=0, high=self.noise_steps, size=(batch_size,), device=self.device)

    def add_noise(self, x0, t):
       
        sqrt_alpha_hat_t = self.sqrt_alpha_hat[t]
        sqrt_1_minus_alpha_hat_t = self.sqrt_1_minus_alpha_hat[t]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_hat_t.shape) < len(x0.shape):
            sqrt_alpha_hat_t = sqrt_alpha_hat_t.unsqueeze(-1)
            sqrt_1_minus_alpha_hat_t = sqrt_1_minus_alpha_hat_t.unsqueeze(-1)
        
        epsilon = torch.randn_like(x0)
        x_t = sqrt_alpha_hat_t * x0 + sqrt_1_minus_alpha_hat_t * epsilon
        
        return x_t, epsilon

    @torch.no_grad()
    def sample(self, model, n_samples):
        
        model.eval()
        

        x = torch.randn((n_samples, 3, 64, 64), device=self.device)
        
       # Reverse diffusion process
        for t in reversed(range(1, self.noise_steps)):
            t_tensor = torch.full((n_samples,), t, dtype=torch.long, device=self.device)
            
            predicted_noise = model(x, t_tensor)
            
            
            sqrt_alpha_t = torch.sqrt(self.alpha[t])
            sqrt_alpha_hat_t = self.sqrt_alpha_hat[t]
            sqrt_1_minus_alpha_hat_t = self.sqrt_1_minus_alpha_hat[t]
            
          
            sqrt_alpha_t = sqrt_alpha_t.view(1, 1, 1, 1)
            sqrt_alpha_hat_t = sqrt_alpha_hat_t.view(1, 1, 1, 1)
            sqrt_1_minus_alpha_hat_t = sqrt_1_minus_alpha_hat_t.view(1, 1, 1, 1)
        
            mean = (1 / sqrt_alpha_t) * (x - ((1 - self.alpha[t]) / sqrt_1_minus_alpha_hat_t) * predicted_noise)
         #   
            
            if t > 1:
                alpha_hat_t_prev = self.alpha_hat[t - 1]
                posterior_variance = self.beta[t] * (1 - alpha_hat_t_prev) / (1 - self.alpha_hat[t])
                std = torch.sqrt(posterior_variance)
                noise = torch.randn_like(x)
                x = mean + std.view(1, 1, 1, 1) * noise
            else:
                x = mean
        
        x=torch.clamp(x, -1, 1) 
        return x
def visualize_forward(self, x0, steps=[0, 200, 400, 600, 800, 999]):
   
    images = []
    for step in steps:
        t = torch.tensor([step], device=self.device)
        x_t, _ = self.add_noise(x0, t)
        images.append(x_t)
    return images
# Train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from dataset import AnimalDataset
from Diffusion import Diffusion
from U_NET import DiffusionUNet
# Custom MSE loss function for training
class customMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        return torch.mean((pred - target) ** 2)
DATASET_PATH = "animal_data"
# Training parameters
BATCH_SIZE = 16
epochs = 200
LEARNING_RATE = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset= AnimalDataset(DATASET_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = DiffusionUNet().to(device)
diffusion = Diffusion(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = customMSELoss()
os.makedirs("Results", exist_ok=True)
os.makedirs("Models", exist_ok=True)
best_loss = float("inf")
loss_history = []
best_epoch = 1
patience = 20
counter = 0
# Training loop
for epoch in range(epochs):
        total_loss = 0
        for images in dataloader:
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0])
            noisy_images, noise = diffusion.add_noise(images, t)
            predicted_noise = model(noisy_images, t)
            loss = criterion(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        # Save best model based on loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            counter = 0
            torch.save(model.state_dict(), "Models/best_model.pth") 
            print("  → New best model saved!")
            # Early stopping logic
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                 generated = diffusion.sample(model, n_samples=4)
            model.train()
            generated = (generated.clamp(-1, 1) + 1) / 2
            save_image(generated, f"Results/generated_epoch_{epoch+1}.png")
            torch.save(model.state_dict(), f"Models/model_epoch_{epoch+1}.pth")

print("Training completed.")    
# Plot and save loss graph
print("\nCreating loss graph...")

plt.figure(figsize=(12, 6))
plt.plot(loss_history, linewidth=2, color='#2E86AB', label='Training Loss')

plt.axvline(x=best_epoch-1, color='red', linestyle='--', linewidth=2, label=f'Best Model (Epoch {best_epoch})')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()

# Save the graph
loss_graph_path = "Results/training_loss_graph.png"
plt.savefig(loss_graph_path, dpi=300, bbox_inches='tight')
print(f"✓ Loss graph saved to {loss_graph_path}")

# Also save as PDF
pdf_path = "Results/training_loss_graph.pdf"
plt.savefig(pdf_path, bbox_inches='tight')
print(f"✓ Loss graph saved to {pdf_path}")

plt.close()

# Print statistics
print(f"\nTraining Statistics:")
print(f"  Initial Loss: {loss[0]:.4f}")
print(f"  Final Loss: {loss[-1]:.4f}")
print(f"  Best Loss: {best_loss:.4f} (Epoch {best_loss})")
print(f"  Total Epochs: {len(loss)}")
print(f"  Loss Reduction: {loss[0] - loss[-1]:.4f}")
#test_reconstruction.py


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

dataset = AnimalDataset("animal_data", max_images_per_class=4, num_classes=5)
dataloader = DataLoader(dataset, batch_size=1)
real_image = next(iter(dataloader)).to(device)

diffusion = Diffusion(device=device)

print("Reconstructing image from noise...\n")

# Add high noise to real image
t_noisy = 500  
t_tensor = torch.full((1,), t_noisy, dtype=torch.long, device=device)

noisy_image, _ = diffusion.add_noise(real_image, t_tensor)

print(f"Real image range: [{real_image.min():.3f}, {real_image.max():.3f}]")
print(f"Noisy image at t={t_noisy} range: [{noisy_image.min():.3f}, {noisy_image.max():.3f}]")
# Save original and noisy images
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
print("  - Results/original.png")
print("  - Results/noisy_t500.png")
print("  - Results/reconstructed.png")
print("\nIf reconstruction ≈ original, the reverse process works!")
# Visualize_noise.py
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from Diffusion import Diffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diffusion = Diffusion(device=device)
img_path = "animal_data/bear/bear_1_2.jpg"  
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

img = Image.open(img_path).convert("RGB")
x0 = transform(img).unsqueeze(0).to(device)  # [1, 3, 64, 64]

timesteps = [0, 50, 100, 200, 400, 600, 800, 999]

fig, axes = plt.subplots(1, len(timesteps), figsize=(20, 3))
fig.suptitle('Noise Progression - MSDS25008', fontsize=13, fontweight='bold')

for i, t in enumerate(timesteps):
    if t == 0:
    
        img_show = x0.squeeze(0)
    else:
        t_tensor = torch.tensor([t], device=device)
        x_t, _ = diffusion.add_noise(x0, t_tensor)
        img_show = x_t.squeeze(0)
 # Denormalize [-1,1] -> [0,1]
    img_show = (img_show.clamp(-1, 1) + 1) / 2
    img_show = img_show.permute(1, 2, 0).cpu().numpy()

    axes[i].imshow(img_show)
    axes[i].set_title(f't={t}', fontsize=9)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("Results/noise_progression.png", dpi=200, bbox_inches='tight')
plt.show()
print("Saved to Results/noise_progression.png")
# test_single_sample.py

import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
import torch
from torchvision.utils import save_image
from dataset import AnimalDataset
from U_NET import DiffusionUNet
from Diffusion import Diffusion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("Results", exist_ok=True)

# Load best model
print("Loading model...")
model_path = "Models/best_model.pth"
# Check if model file exists
if not os.path.exists(model_path):
    print(f"Error: {model_path} not found!")
    print("Available files:")
    if os.path.exists("Models"):
        for f in os.listdir("Models"):
            print(f"  - {f}")
    exit()
    # Load model
model = DiffusionUNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()
print("✓ Model loaded!\n")

    
   # Initialize diffusion
diffusion = Diffusion(device=device)
print("Generating images...")
with torch.no_grad():
    generated = diffusion.sample(model, n_samples=4)
generated = (generated.clamp(-1, 1) + 1) / 2
# Save generated images
output_path = "Results/generated_sample.png"
save_image(generated, output_path)
print(f"✓ Saved to {output_path}")
grid = make_grid(generated, nrow=2)
grid_np = grid.permute(1, 2, 0).cpu().numpy()
plt.figure(figsize=(8, 8))
plt.imshow(grid_np)
plt.axis('off')
plt.title('Generated Images')
plt.savefig("Results/generated_grid.png", dpi=150, bbox_inches='tight')
plt.show()
print("✓ Grid saved!")

print("\n Complete!")