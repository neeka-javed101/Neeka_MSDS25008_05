import torch
import torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
           
        )
    def forward(self, x):
            return self.encoder(x)
class UNetDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.down1=DoubleConv(3,64)
                self.pool1=nn.MaxPool2d(2)  
                self.down2=DoubleConv(64,128)
                self.pool2=nn.MaxPool2d(2)  
                self.midle=DoubleConv(128,256)
                self.up1=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
                self.up_conv1=DoubleConv(256,128)
                self.up2=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
                self.up_conv2=DoubleConv(128,64)
                self.final_conv=nn.Conv2d(64,3,kernel_size=1)
                self.time_embedding=nn.Embedding(1000,256)
            def forward(self,x,t):
            
                d1=self.down1(x)
                d2=self.down2(self.pool1(d1))
                middle=self.midle(self.pool2(d2))
                t_emb=self.time_embedding(t)
                t_emb=t_emb.view(t_emb.shape[0],t_emb.shape[1],1,1  )
                middle=middle+t_emb
                u1=self.up1(middle)
                
                u1 = torch.cat([u1, d2], dim=1)
                u1 = self.up_conv1(u1)
                u2=self.up2(u1)
                u2=torch.cat([u2,d1],dim=1)
                u2=self.up_conv2(u2)
                return self.final_conv(u2)