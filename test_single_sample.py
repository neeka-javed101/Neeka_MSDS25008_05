import torch
from torchvision.utils import save_image
from dataset import AnimalDataset
from U_NET import UNetDecoder
from Diffusion import Diffusion
device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model=UNetDecoder().to(device)
model.load_state_dict(torch.load("Models/model_epoch_50.pth", map_location=device))
model.eval()
diffusion = Diffusion().to(device)
with torch.no_grad():
    generated_images = diffusion.sample(model,1)
    generated_images = (generated_images.clamp(-1, 1) + 1) / 2
    save_image(generated_images, "Results/generated_sample.png")
