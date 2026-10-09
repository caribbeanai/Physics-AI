import torch
import torch.nn.functional as F
from .physics import normalized_physical_consistency,interface_consistency

def total_loss(pred,truth,lat_grid,weights=(1.0,0.05,0.02,0.02),block=16,**physics_kw):
    wr,wp,wi,wm=weights
    rec=F.mse_loss(pred,truth)
    npc=normalized_physical_consistency(pred,truth,lat_grid,**physics_kw)
    phys=torch.stack(list(npc.values())).mean()
    inter=interface_consistency(pred,block)
    pred2=F.avg_pool3d(pred,(1,2,2)); truth2=F.avg_pool3d(truth,(1,2,2))
    lat2=lat_grid[...,::2,::2]
    npc2=normalized_physical_consistency(pred2,truth2,lat2,dlat_deg=physics_kw.get("dlat_deg",0.25)*2,dlon_deg=physics_kw.get("dlon_deg",0.25)*2,dt_s=physics_kw.get("dt_s",21600.0))
    ms=torch.stack(list(npc2.values())).mean()
    return wr*rec+wp*phys+wi*inter+wm*ms,{"rec":rec,"phys":phys,"interface":inter,"multiscale":ms,**npc}
