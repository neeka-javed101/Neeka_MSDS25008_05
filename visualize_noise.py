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