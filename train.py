import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import AnimalDataset
from Diffusion import Diffusion
from U_NET import UNetDecoder
class customMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        return torch.mean((pred - target) ** 2)
DATASET_PATH = "animal_data"
dataset = AnimalDataset(DATASET_PATH)
BATCH_SIZE = 16
epochs = 50
LEARNING_RATE = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset= AnimalDataset(DATASET_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = UNetDecoder().to(device)
diffusion = Diffusion(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = customMSELoss()
os.makedirs("Results", exist_ok=True)
os.makedirs("Models", exist_ok=True)
best_loss = float("inf")
patience = 5
counter = 0
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
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            torch.save(model.state_dict(), "Models/best_model.pth") 
            print("  → New best model saved!")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        if (epoch + 1) % 10 == 0:
            generated = (generated.clamp(-1, 1) + 1) / 2
            save_image(generated, f"Results/generated_epoch_{epoch+1}.png")
            torch.save(model.state_dict(), f"Models/model_epoch_{epoch+1}.pth")
