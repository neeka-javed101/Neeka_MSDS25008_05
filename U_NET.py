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
               