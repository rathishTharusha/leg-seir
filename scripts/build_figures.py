"""Draw diagrams and charts using the original saved experiment results."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import torch

ROOT=Path(__file__).resolve().parent; PKG=ROOT.parent
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--evidence-dir',type=Path,default=PKG/'artifacts/verified')
parser.add_argument('--output-dir',type=Path,default=PKG/'figures/publication')
args=parser.parse_args()
D=args.evidence_dir; OUT=args.output_dir; OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
COL={'ink':'#253d4c','local':'#e6eef5','novel':'#dff1ea','seir':'#fff0db','lagged':'#008474','mlp':'#5a63aa','persist':'#8b3d42'}
data_records=[]
def save(fig,name):
    for ext in ['pdf','svg','png']: fig.savefig(OUT/f'{name}.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)
def panel(w,h):
    fig,ax=plt.subplots(figsize=(w,h));ax.set(xlim=(0,w),ylim=(0,h));ax.axis('off');return fig,ax
def block(ax,x,y,w,h,label,fill='local',size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04,rounding_size=.06',facecolor=COL.get(fill,fill),edgecolor=COL['ink'],lw=1.1))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,color=COL['ink'])
def node(ax,x,y,label):
    ax.add_patch(Circle((x,y),.20,facecolor='white',edgecolor=COL['ink'],lw=1.2));ax.text(x,y,label,ha='center',va='center',fontsize=15)
def arr(ax,pts,dashed=False,color=None,label=None):
    c=color or COL['ink']
    for a,b in zip(pts[:-2],pts[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color=c,lw=1.1,ls='--' if dashed else '-')
    ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle='-|>',mutation_scale=10,lw=1.1,color=c,linestyle='--' if dashed else '-'))
    if label:ax.text(label[0],label[1],label[2],fontsize=9,ha='center',va='center',bbox={'facecolor':'white','edgecolor':'none','pad':1})

fig,ax=panel(18,6.3)
ax.text(.1,6.0,'(a) LEG-SEIR: shared lag selection for neural messages and mechanistic pressure',fontsize=14,weight='bold')
for k in range(3):
    ax.add_patch(Rectangle((.15+k*.13,3.65+k*.12),.9,1,facecolor='white',edgecolor=COL['ink']))
    for j in range(1,4):ax.plot([.15+k*.13,.15+k*.13+.9],[3.65+k*.12+j*.25]*2,color='#a9bdc7',lw=.5)
ax.text(.8,3.13,'Case window C\nN districts × L reports',ha='center',fontsize=10)
block(ax,1.65,3.9,1.4,.65,'Log + scale',size=10)
block(ax,3.5,3.8,1.7,.85,'Shared MLP\n'+r'$\phi$',size=11)
arr(ax,[(1.35,4.22),(1.65,4.22)]);arr(ax,[(3.05,4.22),(3.5,4.22)])
block(ax,6,4.6,1.65,.65,'Local readout',size=10)
arr(ax,[(5.2,4.22),(5.6,4.22),(5.6,4.92),(6,4.92)])
block(ax,3.5,1.65,2.15,1.25,'Learned graph A\n+ lag Softmax w',fill='novel')
for x,y in [(3.78,1.83),(4.0,2.02),(4.2,1.84)]:ax.add_patch(Circle((x,y),.035,color=COL['lagged']))
ax.plot([3.78,4,4.2],[1.83,2.02,1.84],color=COL['lagged'],lw=.8)
arr(ax,[(2.35,3.9),(2.35,2.25),(3.5,2.25)],label=(2.4,2.8,'X'))
block(ax,6.05,1.95,1.6,.65,'Shared MLP',size=10)
arr(ax,[(5.65,2.25),(6.05,2.25)],label=(5.86,2.65,'q'))
node(ax,8.2,2.27,'−');arr(ax,[(7.65,2.27),(8,2.27)])
arr(ax,[(5.2,4.22),(8.2,4.22),(8.2,2.47)],label=(6.8,4.22,'local h'))
node(ax,9.4,2.27,'×');block(ax,8.7,3.0,1.4,.55,'Sigmoid gate',size=9)
block(ax,8.55,2.03,.45,.48,r'$W_m$',size=9)
arr(ax,[(8.4,2.27),(8.55,2.27)]);arr(ax,[(9.0,2.27),(9.2,2.27)]);arr(ax,[(9.4,3.0),(9.4,2.47)])
node(ax,10.4,2.27,'+');arr(ax,[(9.6,2.27),(10.2,2.27)])
arr(ax,[(8.2,4.22),(10.4,4.22),(10.4,2.47)])
block(ax,11.1,3.7,1.7,.65,'Graph readout',size=10)
arr(ax,[(10.6,2.27),(10.8,2.27),(10.8,4.02),(11.1,4.02)])
block(ax,11.1,2.0,1.7,.65,'Rate head',fill='seir',size=10)
arr(ax,[(10.6,2.27),(11.1,2.27)])
block(ax,10.9,.5,2.1,.75,'SEIR rollout\n+ incidence readout',fill='seir',size=10)
arr(ax,[(11.95,2),(11.95,1.25)],label=(12.4,1.6,r'$\beta$'))
arr(ax,[(4.6,1.65),(4.6,.85),(10.9,.85)],color=COL['lagged'],label=(7.6,.85,'Historical pressure → simulated pressure'))
arr(ax,[(.8,3.65),(.8,.2),(11.4,.2),(11.4,.5)],dashed=True,label=(3,.2,'Raw-count initialization'))
block(ax,13.75,2.0,1.5,2.35,'Horizon\nSoftmax\n\n4-expert\nweighted sum',fill='novel',size=11)
arr(ax,[(7.65,4.92),(14.5,4.92),(14.5,4.35)])
arr(ax,[(12.8,4.02),(13.75,4.02)])
arr(ax,[(13,.85),(13.35,.85),(13.35,2.3),(13.75,2.3)])
arr(ax,[(2.35,4.55),(2.35,5.55),(15.55,5.55),(15.55,4.0),(15.25,4.0)],dashed=True,label=(10,5.55,'Persistence bypass p'))
block(ax,16,2.7,1.7,.85,'Inverse scale\n+ floor at zero',size=10)
arr(ax,[(15.25,3.15),(16,3.15)])
ax.text(16.85,2.35,r'$\hat Y\in\mathbb{R}^{N\times H}$',ha='center',fontsize=12)
ax.text(.15,5.55,'Past observed reports only',fontsize=9)
save(fig,'architecture')

fig,ax=panel(16,4.7)
ax.text(.1,4.45,'(b) Imported-pressure bridge inside the decoder',fontsize=14,weight='bold')
block(ax,.2,2.9,2.1,.65,'Lagged raw pressure a',fill='novel',size=10)
block(ax,.2,1.4,2.1,.65,'Simulated neighbours',fill='seir',size=10)
node(ax,3.5,3.22,'×');node(ax,3.5,1.72,'×')
arr(ax,[(2.3,3.22),(3.3,3.22)]);arr(ax,[(2.3,1.72),(3.3,1.72)])
ax.text(3.5,3.7,r'$d_k$',ha='center',fontsize=12);ax.text(3.5,2.2,r'$1-d_k$',ha='center',fontsize=12)
node(ax,4.65,2.5,'+');arr(ax,[(3.7,3.22),(4.65,3.22),(4.65,2.7)]);arr(ax,[(3.7,1.72),(4.65,1.72),(4.65,2.3)])
block(ax,5.5,2.15,1.8,.7,'Local/import\ngated blend',fill='novel',size=10)
arr(ax,[(4.85,2.5),(5.5,2.5)])
block(ax,5.5,.7,1.8,.55,'Current local I',size=10);arr(ax,[(6.4,1.25),(6.4,2.15)])
node(ax,8.1,2.5,'×');arr(ax,[(7.3,2.5),(7.9,2.5)])
ax.text(8.1,3.08,r'$\beta_{i,h}$',ha='center',fontsize=12)
block(ax,8.8,2.12,2.2,.75,'Conservative\ndaily SEIR step',fill='seir',size=10);arr(ax,[(8.3,2.5),(8.8,2.5)])
for x,t in [(11.8,'S'),(12.8,'E'),(13.8,'I'),(14.8,'R')]:
    block(ax,x,2.15,.65,.65,t,fill='seir',size=12)
    if x<14.8:arr(ax,[(x+.65,2.48),(x+1,2.48)])
arr(ax,[(11,2.5),(11.8,2.5)])
arr(ax,[(14.15,2.15),(14.15,.32),(1.2,.32),(1.2,1.4)],dashed=True)
ax.text(10,.55,'State feedback; repeat daily',fontsize=10)
ax.text(.2,3.94,r'$d_k=\exp(-k/\tau)$'+'  (fixed decay)',fontsize=11)
ax.text(11.75,1.6,'Sum E → I flow\nper horizon',fontsize=10)
save(fig,'pressure_bridge')

s=pd.read_csv(D/'scores.csv'); final=s[(s.phase=='final')&(s.split=='test')&~s.sensitivity.astype(bool)]
labels={'lagged_full':'LEG-SEIR','shared_adapter':'Shared adapter','identity_mlp':'Identity MLP','horizon_full':'Horizon mixture','observer_full':'State observer','plain_graph':'Plain graph','uniform_mixture':'Uniform mixture','lagged_seir_off':'LEG-SEIR: neural replacement','lagged_graph_off':'LEG-SEIR: graph removed','local_mlp':'Local MLP','local_seir':'Local SEIR'}
pers=np.load(D/'persistence_final_test.npz');prmse=float(np.sqrt(np.mean((pers['prediction']-pers['truth'])**2)))
rows=[]
for name,g in final.groupby('name'):
    rows.append({'name':name,'label':labels.get(name,name),'RMSE':g.RMSE.mean(),'SD':g.RMSE.std(),'MAE':g.MAE.mean(),'sMAPE':g.sMAPE.mean(),'parameters':int(g.parameters.iloc[0]),'lr':g.lr.iloc[0]})
summary=pd.DataFrame(rows).sort_values('RMSE');summary.to_csv(OUT/'primary_results.csv',index=False)
chosen=['lagged_full','shared_adapter','identity_mlp','horizon_full','observer_full','plain_graph','local_mlp']
fig,(ax,bx)=plt.subplots(1,2,figsize=(10,3.65),gridspec_kw={'width_ratios':[1.2,1]})
for i,name in enumerate(chosen):
    vals=final[final.name==name].RMSE.to_numpy();color=COL['lagged'] if name=='lagged_full' else COL['mlp'] if name=='identity_mlp' else '#516879'
    ax.errorbar(vals.mean(),i,xerr=vals.std(ddof=1),fmt='o',color=color,capsize=3)
    ax.scatter(vals,i+np.linspace(-.08,.08,len(vals)),s=9,color=color,alpha=.45)
ax.axvline(prmse,color=COL['persist'],ls='--',label=f'Persistence {prmse:.2f}')
ax.set(yticks=range(len(chosen)),yticklabels=[labels[n] for n in chosen],xlabel='Count RMSE ↓',title='(a) Primary comparison');ax.invert_yaxis();ax.legend(fontsize=8,loc='lower right');ax.grid(axis='x',alpha=.15)
control=['lagged_full','lagged_seir_off','lagged_graph_off']
for i,name in enumerate(control):
    vals=final[final.name==name].RMSE.to_numpy();bx.scatter(np.arange(5),vals,label=labels[name],s=28)
bx.axhline(prmse,color=COL['persist'],ls='--',label='Persistence');bx.set(xlabel='Training seed',ylabel='Count RMSE ↓',xticks=range(5),title='(b) Matched removals',ylim=(77.8,93.5));bx.legend(fontsize=7,loc='upper left');bx.grid(alpha=.15)
fig.tight_layout();save(fig,'results_comparison')

freeze=json.loads((D/'learning_rate_freeze.json').read_text())['primary']
fig,axes=plt.subplots(1,2,figsize=(10,3))
dev=s[(s.phase=='development')&(s.split=='val')]
for name in ['lagged_full','identity_mlp','shared_adapter','horizon_full','observer_full']:
    v=[]
    for year in [2023,2025]:v.append(dev[(dev.name==name)&(dev.lr==freeze[name])&(dev.year==year)].skill.mean()*100)
    axes[0].plot([0,1],v,'o-',label=labels[name],lw=1.3)
axes[0].set(xticks=[0,1],xticklabels=['2023 validation','2025 validation'],ylabel='RMSE reduction vs persistence (%)',title='(a) Development evidence');axes[0].legend(fontsize=7,ncol=2);axes[0].grid(alpha=.15)
for name in ['lagged_full','identity_mlp','shared_adapter']:
    g=final[final.name==name];means=g[['RMSE_h1','RMSE_h2','RMSE_h3']].mean().to_numpy();sd=g[['RMSE_h1','RMSE_h2','RMSE_h3']].std().to_numpy()
    axes[1].errorbar([1,2,3],means,yerr=sd,fmt='o-',capsize=3,label=labels[name])
p_h=np.sqrt(np.mean((pers['prediction']-pers['truth'])**2,axis=(0,1)))
axes[1].plot([1,2,3],p_h,'--',color=COL['persist'],label='Persistence');axes[1].set(xticks=[1,2,3],xlabel='Report-observation horizon',ylabel='Count RMSE ↓',title='(b) 2026 horizons');axes[1].legend(fontsize=7);axes[1].grid(alpha=.15)
fig.tight_layout();save(fig,'development_horizons')

fig,ax=plt.subplots(figsize=(5.8,1.8));ax.set(xlim=(2012.8,2027.2),ylim=(-.6,3.6));ax.set_yticks([3,2,1],['Develop. A','Develop. B','Final'])
for y,end,val in [(3,2022,2023),(2,2024,2025),(1,2024,2025)]:
    ax.barh(y,end+1-2013,left=2013,height=.38,color='#6c90ad');ax.barh(y,1,left=val,height=.38,color='#d8ac58')
ax.barh(1,1,left=2026,height=.38,color='#569b89')
ax.text(2017,.1,'Train',color='#4c7492',fontsize=10);ax.text(2022,.1,'Validate',color='#906b26',fontsize=10);ax.text(2026,.1,'Test',color='#387b6b',fontsize=10,ha='center')
ax.set_xticks([2013,2022,2024,2026]);ax.spines[['top','right','left']].set_visible(False);fig.tight_layout();save(fig,'evaluation_timeline')

case=pd.read_csv(D/'cases_25.csv');n=np.load(D/'final_lagged_full_lr0.003_seed0_test.npz');print('NPZ',[(k,n[k].shape) for k in n.files]);idx=n['idx'];print('COLUMNS',list(case.columns));print('PERSISTENCE',prmse)
fig,axes=plt.subplots(1,3,figsize=(11,3),sharey=True)
ensemble={}
for name in ['lagged_full','identity_mlp','shared_adapter']:
    rr=final[final.name==name];ensemble[name]=np.mean([np.load(D/(token+'_test.npz'))['prediction'] for token in rr.token],axis=0)
for h,ax in enumerate(axes):
    target=idx+h; order=np.argsort(target);target=target[order]
    # Leave a gap where a report is missing; do not draw a line as if it was observed.
    x=target.astype(float);gaps=np.r_[False,np.diff(target)>1]
    def segmented(y):
        xx=[];yy=[]
        for k,(a,b) in enumerate(zip(x,y[order])):
            if gaps[k]:xx.append(np.nan);yy.append(np.nan)
            xx.append(a);yy.append(b)
        return np.array(xx),np.array(yy)
    for name,y,c,ls in [('Observed',n['truth'][:,:,h].sum(1),'#253d4c','-'),('Persistence',pers['prediction'][:,:,h].sum(1),COL['persist'],'--'),('LEG-SEIR',ensemble['lagged_full'][:,:,h].sum(1),COL['lagged'],'-'),('Identity MLP',ensemble['identity_mlp'][:,:,h].sum(1),COL['mlp'],':')]:
        a,b=segmented(y);ax.plot(a,b,ls,marker='o',ms=2.3,lw=1.1,color=c,label=name)
    ax.set(title=f'Horizon {h+1}',xlabel='Target report slot');ax.grid(alpha=.15)
axes[0].set_ylabel('National reported total');axes[-1].legend(fontsize=7);fig.tight_layout();save(fig,'case_study')

ck=torch.load(D/'final_lagged_full_lr0.003_seed0.pt',map_location='cpu',weights_only=False)
sd=ck['state_dict'];score=(sd['left']@sd['right'].T);score.fill_diagonal_(-torch.inf);keep=score.topk(5,dim=-1).indices;mask=torch.zeros_like(score,dtype=torch.bool).scatter(1,keep,True);A=torch.softmax(score.masked_fill(~mask,-torch.inf),-1).numpy()
fig,axes=plt.subplots(1,3,figsize=(11,3.3),gridspec_kw={'width_ratios':[1.3,.7,1]})
im=axes[0].imshow(A,cmap='Blues',vmin=0,vmax=1);axes[0].set(title='(a) Learned adjacency (seed 0)',xlabel='Sending district index',ylabel='Receiving district index');fig.colorbar(im,ax=axes[0],fraction=.046)
lag=torch.softmax(sd['lag_logits'],-1).numpy();im=axes[1].imshow(lag,aspect='auto',cmap='Greens',vmin=0,vmax=1);axes[1].set(title='(b) Lag weights',xticks=[0,1,2],xticklabels=['Oldest','Middle','Latest'],xlabel='Historical report position');fig.colorbar(im,ax=axes[1],fraction=.046)
weights=torch.softmax(sd['router_logits'],-1).numpy();im=axes[2].imshow(weights,cmap='Oranges',aspect='auto',vmin=0,vmax=1);axes[2].set(title='(c) Horizon mixture',xticks=[0,1,2,3],xticklabels=['Persist.','Local','Graph','SEIR'],yticks=[0,1,2],yticklabels=['h1','h2','h3']);fig.colorbar(im,ax=axes[2],fraction=.046)
fig.tight_layout();save(fig,'learned_diagnostics')

fig,axes=plt.subplots(1,2,figsize=(9,3))
for seed in range(5):
    c=pd.read_csv(D/f'final_lagged_full_lr0.003_seed{seed}_curve.csv');axes[0].plot(c.epoch,c.train_loss,label=f'Seed {seed}');axes[1].plot(c.epoch,c.validation_RMSE)
axes[0].set(xlabel='Epoch',ylabel='Standardized-log training MSE',title='(a) Objective');axes[1].set(xlabel='Epoch',ylabel='2025 validation count RMSE',title='(b) Checkpoint selection');axes[0].legend(fontsize=7);[a.grid(alpha=.15) for a in axes];fig.tight_layout();save(fig,'training_curves')

# Keep all table rows, including worse results and extra learning-rate checks.
s.to_csv(OUT/'all_scored_records.csv',index=False)
stats={}
for year in [2023,2025]:
    z=np.load(D/f'persistence_dev_{year}.npz');stats[str(year)]=float(np.sqrt(np.mean((z['prediction']-z['truth'])**2)))
stats['2026']=prmse
stats['test_origins']=len(idx)
stats['lagged_parameters']=int(final[final.name=='lagged_full'].parameters.iloc[0])
stats['lagged_mean_horizons']=final[final.name=='lagged_full'][['RMSE_h1','RMSE_h2','RMSE_h3']].mean().tolist()
stats['source_files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'scores.csv',D/'learning_rate_freeze.json']}
(OUT/'FIGURE_DATA_AUDIT.json').write_text(json.dumps(stats,indent=2))
alternative=pd.read_csv(D/'cases_25_printed_repeat_sensitivity.csv').iloc[:,:25].to_numpy()
raw=case.iloc[:,:25].to_numpy();years=case.clinical_year.to_numpy();mask=years==2026
fig,(ax,bx)=plt.subplots(1,2,figsize=(10,3))
unique=sorted(set(years));finite=np.isfinite(raw).all(1)
ax.bar(unique,[np.sum(finite&(years==y)) for y in unique],label='Finite primary report rows',color='#598b9c')
ax.bar(unique,[np.sum(~finite&(years==y)) for y in unique],bottom=[np.sum(finite&(years==y)) for y in unique],label='Missing/unresolved',color='#d5a46b')
ax.set(xlabel='Report-period year',ylabel='Report slots',title='(a) Source coverage');ax.legend(fontsize=7);ax.set_xticks(unique[::2])
xx=np.flatnonzero(mask);pvals=np.where(finite,raw.sum(1),np.nan);afinite=np.isfinite(alternative).all(1);avals=np.where(afinite,alternative.sum(1),np.nan)
bx.plot(xx,avals[mask],'o--',ms=3,lw=1,color='#d5a46b',label='Printed-repeat alternative (data only)')
bx.plot(xx,pvals[mask],'o-',ms=3,lw=1,color=COL['ink'],label='Conservative primary series')
for pos in xx[(~finite&afinite)[mask]]:bx.axvspan(pos-.4,pos+.4,color='#d5a46b',alpha=.2)
bx.set(xlabel='2026 report slot',ylabel='National reported total',title='(b) Unresolved-source context');bx.legend(fontsize=7);bx.grid(alpha=.15)
fig.tight_layout();save(fig,'source_coverage')
print('AUDIT',json.dumps(stats,indent=2))
