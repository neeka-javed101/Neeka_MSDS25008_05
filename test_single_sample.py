
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

print("\n✅ Complete!")