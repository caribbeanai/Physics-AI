import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self,c=64):
        super().__init__(); self.f=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.ReLU(),nn.Conv2d(c,c,3,padding=1))
    def forward(self,x): return x+self.f(x)

class MultiVarEDSR(nn.Module):
    def __init__(self,in_ch=15,out_ch=15,width=64,blocks=8,scale=4):
        super().__init__()
        self.head=nn.Conv2d(in_ch,width,3,padding=1)
        self.body=nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.tail=nn.Sequential(nn.Conv2d(width,out_ch*scale*scale,3,padding=1),nn.PixelShuffle(scale))
    def forward(self,x):
        h=self.head(x); h=self.body(h); return self.tail(h)

class DLegosPISR(nn.Module):
    """Local block encoding + neighbour message passing + SR decoding."""
    def __init__(self,in_ch=15,out_ch=15,width=64,blocks=8,scale=4,block_px=8):
        super().__init__(); self.block_px=block_px
        self.encoder=nn.Sequential(nn.Conv2d(in_ch,width,3,padding=1),nn.GELU(),*[ResidualBlock(width) for _ in range(blocks//2)])
        self.message=nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.GELU(),nn.Conv2d(width,width,3,padding=1))
        self.decoder=nn.Sequential(*[ResidualBlock(width) for _ in range(blocks//2)],nn.Conv2d(width,out_ch*scale*scale,3,padding=1),nn.PixelShuffle(scale))
    def forward(self,x):
        z=self.encoder(x)
        p=F.avg_pool2d(z,self.block_px,self.block_px)
        m=self.message(p)
        m=F.interpolate(m,size=z.shape[-2:],mode="nearest")
        return self.decoder(z+m)
