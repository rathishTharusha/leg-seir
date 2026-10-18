def conservative_step(state,force,gamma=1/7,omega=.1):
    s,e,i,r=state.unbind(-1)
    u=s*(-torch.expm1(-force))
    v=e*(-math.expm1(-omega))
    w=i*(-math.expm1(-gamma))
    return torch.stack([s-u,e+u-v,i+v-w,r+w],-1),v

def mechanistic_forecast(rawx,raw,feedback=False,gamma=1/7):
    scale=torch.as_tensor(POP,device=rawx.device,dtype=rawx.dtype)[None]/11
    i=(rawx[:,:,-1]/scale).clamp(1e-6,.5)
    e=(rawx[:,:,-2]/scale).clamp(1e-6,.5)
    s=torch.full_like(i,.318)
    r=(1-s-e-i).clamp_min(0)
    state=torch.stack([s,e,i,r],-1)
    state=state/state.sum(-1,keepdim=True)
    prediction=[]
    for h in range(HORIZON):
        # Separate parameterizations: old neural force versus neural beta.
        rate=torch.exp((raw[:,:,h]-(1.1 if feedback else 9.36)).clamp(*((-9,1.1) if feedback else (-18,-1.95))))
        total=torch.zeros_like(i)
        for _ in range(7):
            force=rate*state[:,:,2] if feedback else rate
            state,onset=conservative_step(state,force,gamma=gamma)
            total=total+onset
        prediction.append(total*scale)
    return torch.stack(prediction,-1)

