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