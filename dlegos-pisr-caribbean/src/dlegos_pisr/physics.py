import torch
R_D=287.05; G=9.80665; CP=1004.0; OMEGA=7.2921159e-5; EARTH_R=6_371_000.0

def sat_vapor_pressure(td_k):
    td_c=td_k-273.15
    return 611.2*torch.exp(17.67*td_c/(td_c+243.5))

def specific_humidity_from_dewpoint(td_k,p_pa):
    e=sat_vapor_pressure(td_k)
    return 0.622*e/(p_pa-0.378*e).clamp_min(1.0)

def virtual_temperature(t_k,td_k,p_pa):
    q=specific_humidity_from_dewpoint(td_k,p_pa)
    return t_k*(1+0.61*q)

def air_density(t_k,td_k,p_pa):
    return p_pa/(R_D*virtual_temperature(t_k,td_k,p_pa).clamp_min(150.0))

def grad_xy(x,lat_deg,dlat_deg,dlon_deg):
    lat=torch.deg2rad(lat_deg)
    dy=EARTH_R*torch.deg2rad(torch.as_tensor(dlat_deg,device=x.device,dtype=x.dtype))
    dx=EARTH_R*torch.cos(lat).clamp_min(0.1)*torch.deg2rad(torch.as_tensor(dlon_deg,device=x.device,dtype=x.dtype))
    gx=(torch.roll(x,-1,-1)-torch.roll(x,1,-1))/(2*dx)
    gy=(torch.roll(x,-1,-2)-torch.roll(x,1,-2))/(2*dy)
    return gx,gy

def physics_terms(state,lat_grid,dt_s=21600.0,dlat_deg=0.25,dlon_deg=0.25):
    T,Td,p,u,v=[state[:,:,i] for i in range(5)]
    rho=air_density(T,Td,p)
    drhodt=(rho[:,2:]-rho[:,:-2])/(2*dt_s)
    fx,_=grad_xy(rho*u,lat_grid,dlat_deg,dlon_deg)
    _,fy=grad_xy(rho*v,lat_grid,dlat_deg,dlon_deg)
    mass=drhodt+fx[:,1:-1]+fy[:,1:-1]
    dudt=(u[:,2:]-u[:,:-2])/(2*dt_s); dvdt=(v[:,2:]-v[:,:-2])/(2*dt_s)
    dpdx,dpdy=grad_xy(p,lat_grid,dlat_deg,dlon_deg)
    f=2*OMEGA*torch.sin(torch.deg2rad(lat_grid))
    mom_u=dudt-(f*v)[:,1:-1]+(dpdx/rho)[:,1:-1]
    mom_v=dvdt+(f*u)[:,1:-1]+(dpdy/rho)[:,1:-1]
    dTdt=(T[:,2:]-T[:,:-2])/(2*dt_s); dpdt=(p[:,2:]-p[:,:-2])/(2*dt_s)
    thermo=dTdt-dpdt/(CP*rho[:,1:-1])
    Hs=p/(rho*G); hx,hy=grad_xy(Hs,lat_grid,dlat_deg,dlon_deg)
    hydro=torch.sqrt(hx[:,1:-1]**2+hy[:,1:-1]**2+1e-12)
    return {"hydro":hydro,"mass":mass,"mom_u":mom_u,"mom_v":mom_v,"thermo":thermo}

def normalized_physical_consistency(pred,truth,lat_grid,**kw):
    rp=physics_terms(pred,lat_grid,**kw); rt=physics_terms(truth,lat_grid,**kw); out={}
    for k in rp:
        out[k]=(rp[k]-rt[k]).abs().mean()/rt[k].abs().mean().clamp_min(1e-8)
    out["momentum"]=0.5*(out.pop("mom_u")+out.pop("mom_v"))
    return out

def interface_consistency(state,block=16):
    eps=1e-6; vals=[]; H,W=state.shape[-2:]
    for x in range(block,W,block):
        a=state[...,x-1]; b=state[...,x]
        vals.append((a-b).abs().mean()/((a.abs()+b.abs()).mean()+eps))
    for y in range(block,H,block):
        a=state[...,y-1,:]; b=state[...,y,:]
        vals.append((a-b).abs().mean()/((a.abs()+b.abs()).mean()+eps))
    return torch.stack(vals).mean() if vals else torch.zeros((),device=state.device)
