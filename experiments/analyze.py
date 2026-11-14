import csv,json,math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_experiment import ROOT,RAW,Net,losses,flat_grad,vec,put,proposal

def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def audit_jvp():
    rows=[]
    for freq in [1,3]:
        for seed in [0,1,2]:
            net=Net();net.load_state_dict(torch.load(RAW/'main'/f'initial_f{freq}_s{seed}.pt',weights_only=False)['model'])
            theta=vec(net); params=list(net.named_parameters());xb=torch.tensor([[0.],[1.]])
            def residual(v):
                offset=0;pd={}
                for name,p in params:
                    pd[name]=v[offset:offset+p.numel()].view_as(p);offset+=p.numel()
                return (torch.func.functional_call(net,pd,(xb,))-xb).flatten()
            p,b=losses(net,torch.linspace(0,1,66)[1:-1,None],freq)
            gp,gb=flat_grad(p,net),flat_grad(b,net)
            for opt in ['adam','sgdm']:
                state={'t':0,'m':torch.zeros_like(theta),'v':torch.zeros_like(theta)}
                for weight in [0.,1.,10.,100.]:
                    for scale in [1.,.25,.0625]:
                        d,_=proposal(gp+weight*gb,state,opt,(.001 if opt=='adam' else .0001)*scale)
                        r,jd=torch.func.jvp(residual,(theta,),(d,))
                        linear=float(2*(r*jd).mean());quad=float(jd.square().mean());base=float(r.square().mean())
                        actual=float(residual(theta+d).square().mean())-base
                        rows.append({'freq':freq,'seed':seed,'optimizer':opt,'weight':weight,'scale':scale,'baseline_bc':base,'actual_change':actual,'linear':linear,'quadratic':linear+quad,'linear_error':abs(linear-actual),'quadratic_error':abs(linear+quad-actual),'linear_false_safe':int(base+linear<=1e-4 and base+actual>1e-4),'quadratic_false_safe':int(base+linear+quad<=1e-4 and base+actual>1e-4)})
    save_csv(RAW/'boundary_jvp_audit.csv',rows)
    result={'n_candidates':len(rows),'linear_mae':float(np.mean([r['linear_error'] for r in rows])),'quadratic_mae':float(np.mean([r['quadratic_error'] for r in rows])),'linear_false_safe':sum(r['linear_false_safe'] for r in rows),'quadratic_false_safe':sum(r['quadratic_false_safe'] for r in rows),'actual_unsafe':sum(r['baseline_bc']+r['actual_change']>1e-4 for r in rows)}
    (RAW/'boundary_jvp_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def main():
    groups=[];allrows=[]
    for tag in ['main','extended']:
        data=json.loads((RAW/tag/f'summary_{tag}.json').read_text())
        assert len(data)==48,(tag,len(data))
        for r in data:
            assert r['rule']!='budget' or r['max_bc']<=r['tau']
            assert r['rule']!='strict' or r['bc_uphill']==0
        for freq in [1,3]:
            for opt in ['adam','sgdm']:
                for rule in ['strict','budget','free','vanilla']:
                    rs=[r for r in data if r['freq']==freq and r['optimizer']==opt and r['rule']==rule]
                    g={'bank':tag,'freq':freq,'optimizer':opt,'rule':rule,'n':len(rs)}
                    for key in ['test_pde','relative_l2','bc','seconds','accepted','bc_uphill','bc_violation_steps']:
                        vals=[r[key] for r in rs];g[key+'_mean']=float(np.mean(vals));g[key+'_std']=float(np.std(vals,ddof=1))
                    groups.append(g)
        allrows+= [{'bank':tag,**r} for r in data]
    save_csv(RAW/'aggregate.csv',groups)
    (RAW/'all_results.json').write_text(json.dumps(allrows,indent=2),encoding='utf-8')
    colors={'strict':'#777777','budget':'#0072B2','free':'#D55E00','vanilla':'#009E73'}
    for tag in ['main','extended']:
        fig,axes=plt.subplots(2,4,figsize=(15,7))
        for col,(freq,opt) in enumerate([(1,'adam'),(1,'sgdm'),(3,'adam'),(3,'sgdm')]):
            for rule in colors:
                traces=[]
                for seed in [0,1,2]:
                    with (RAW/tag/f'f{freq}_s{seed}_{opt}_{rule}.csv').open() as f: traces.append(list(csv.DictReader(f)))
                for row,key in enumerate(['pde','bc']):
                    a=np.array([[float(r[key]) for r in tr] for tr in traces])
                    axes[row,col].plot(np.arange(1,a.shape[1]+1),np.median(a,axis=0),label=rule,color=colors[rule])
                    axes[row,col].fill_between(np.arange(1,a.shape[1]+1),a.min(axis=0),a.max(axis=0),color=colors[rule],alpha=.1)
                    axes[row,col].set_yscale('log');axes[row,col].set_xlabel('Attempted updates');axes[row,col].set_ylabel(key+' MSE')
            axes[0,col].set_title(f'k={freq}, {opt}')
            axes[1,col].axhline(1e-4,color='black',ls='--',lw=1)
        axes[0,0].legend(fontsize=8);fig.suptitle(f'{tag}: median and seed range (n=3); exact candidate evaluation')
        fig.tight_layout();fig.savefig(ROOT/'figures'/f'{tag}_trajectories.png',dpi=160);fig.savefig(ROOT/'figures'/f'{tag}_trajectories.pdf');plt.close(fig)
    jvp=audit_jvp()
    print(json.dumps({'aggregate':groups,'jvp':jvp},indent=2))

if __name__=='__main__':main()
