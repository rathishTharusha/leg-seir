def pack(fold,split,cases=CASES):
    ids=fold['idx'][split]
    rawx=np.stack([cases[i-WINDOW:i].T for i in ids]).astype(np.float32)
    rawy=np.stack([cases[i:i+HORIZON].T for i in ids]).astype(np.float64)
    x=(np.log1p(rawx)-fold['mean'])/fold['std']
    y=(np.log1p(rawy)-fold['mean'])/fold['std']
    return {'idx':ids,'x':torch.tensor(x,device=DEVICE),'y':torch.tensor(y,dtype=torch.float32,device=DEVICE),
            'p':torch.tensor(np.repeat(x[:,:,-1:],HORIZON,2),device=DEVICE),
            'rawx':torch.tensor(rawx,device=DEVICE),'truth':rawy}
def counts(z,fold):
    y=np.expm1(np.asarray(z,dtype=np.float64)*fold['std']+fold['mean'])
    assert np.isfinite(y).all(), 'Nonfinite inverse forecast'
    return np.maximum(y,0)
def metrics(pred,true):
    err=pred-true
    den=np.abs(pred)+np.abs(true)
    smape=np.divide(200*np.abs(err),den,out=np.zeros_like(err),where=den>0)
    return {'RMSE':float(np.sqrt(np.mean(err**2))),'MAE':float(np.abs(err).mean()),'sMAPE':float(smape.mean())}
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
def predict_z(model,data):
    model.eval()
    with torch.no_grad():
        return model(data['x'],data['p'],data['rawx']).cpu().numpy()
