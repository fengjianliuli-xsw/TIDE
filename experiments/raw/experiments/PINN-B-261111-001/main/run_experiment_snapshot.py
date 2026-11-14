import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.dirname(__file__), '.mplcache'))
import argparse, copy, csv, hashlib, json, math, platform, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'raw' / 'experiments' / 'PINN-B-261111-001'
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1,24),nn.Tanh(),nn.Linear(24,24),nn.Tanh(),nn.Linear(24,1))
    def forward(self,x): return self.layers(x)

def losses(net,x,freq, graph=True):
    xx=x.detach().requires_grad_(True)
    u=net(xx)
    ux=torch.autograd.grad(u.sum(),xx,create_graph=True)[0]
    uxx=torch.autograd.grad(ux.sum(),xx,create_graph=graph)[0]
    residual=uxx+(freq*math.pi)**2*torch.sin(freq*math.pi*xx)
    pde=(residual**2).mean()
    xb=torch.tensor([[0.],[1.]])
    bc=((net(xb)-xb)**2).mean()
    return pde,bc

def flat_grad(loss,net):
    gs=torch.autograd.grad(loss,tuple(net.parameters()),retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,net.parameters())]).detach()

def vec(net): return torch.nn.utils.parameters_to_vector(net.parameters()).detach().clone()
def put(net,v):
    with torch.no_grad(): torch.nn.utils.vector_to_parameters(v.clone(),net.parameters())

def proposal(g,state,opt,lr):
    t=state['t']+1
    if opt=='adam':
        m=.9*state['m']+.1*g; v=.999*state['v']+.001*g.square()
        d=-lr*(m/(1-.9**t))/(torch.sqrt(v/(1-.999**t))+1e-8)
    else:
        m=.9*state['m']+g; v=state['v']; d=-lr*m
    return d,{'t':t,'m':m,'v':v}

def prepare(seed,freq):
    torch.manual_seed(seed)
    net=Net(); opt=torch.optim.Adam(net.parameters(),lr=.003)
    xb=torch.tensor([[0.],[1.]])
    for _ in range(800):
        opt.zero_grad(); l=((net(xb)-xb)**2).mean(); l.backward(); opt.step()
        if l.item()<1e-10: break
    actual=((net(xb)-xb)**2).mean().item()
    assert actual<1e-6, actual
    return net,actual

def metrics(net,freq):
    x=torch.linspace(0,1,513)[:,None]
    p,b=losses(net,x[1:-1],freq,False)
    with torch.no_grad():
        truth=torch.sin(freq*math.pi*x)+x
        rel=(torch.linalg.vector_norm(net(x)-truth)/torch.linalg.vector_norm(truth)).item()
    return {'test_pde':p.item(),'bc':b.item(),'relative_l2':rel}

def run(seed,freq,opt,rule,steps,tau,initial):
    net=copy.deepcopy(initial); theta=vec(net)
    state={'t':0,'m':torch.zeros_like(theta),'v':torch.zeros_like(theta)}
    x=torch.linspace(0,1,66)[1:-1,None]
    lr=.001 if opt=='adam' else .0001
    hist=[]; accepted=0; uphill=0; violation=0; evals=0; no_common=0; opportunity=0
    start=time.perf_counter()
    for step in range(steps):
        p,b=losses(net,x,freq); pv,bv=p.item(),b.item()
        gp,gb=flat_grad(p,net),flat_grad(b,net)
        candidates=[]
        weights=[1.] if rule=='vanilla' else [0.,1.,10.,100.]
        scales=[1.] if rule=='vanilla' else [1.,.25,.0625]
        for weight in weights:
            for scale in scales:
                d,ns=proposal(gp+weight*gb,state,opt,lr*scale)
                put(net,theta+d)
                pp,bb=losses(net,x,freq,False); pp,bb=pp.item(),bb.item();evals+=1
                if math.isfinite(pp) and math.isfinite(bb):
                    candidates.append((pp,bb,d,ns,weight,scale,float(gb@d),float(gp@d)))
        common=[c for c in candidates if c[0]<pv and c[1]<=bv+1e-12]
        safe=[c for c in candidates if c[0]<pv and c[1]<=tau]
        if not common: no_common+=1
        if safe and (not common or min(c[0] for c in safe)<min(c[0] for c in common)): opportunity+=1
        feasible={'strict':common,'budget':safe,'free':[c for c in candidates if c[0]<pv], 'vanilla':candidates}[rule]
        if feasible:
            c=min(feasible,key=lambda c:c[0]);pp,bb,d,state,w,scale,linb,linp=c
            theta=theta+d;accepted+=1;uphill+=int(bb>bv+1e-12)
        else: pp,bb,w,scale,linb,linp=pv,bv,-1.,0.,0.,0.
        put(net,theta);violation+=int(bb>tau)
        hist.append({'step':step+1,'pde':pp,'bc':bb,'accepted':int(bool(feasible)), 'weight':w,'scale':scale,'linear_bc_change':linb,'linear_pde_change':linp,'actual_bc_change':bb-bv,'actual_pde_change':pp-pv})
    elapsed=time.perf_counter()-start
    summary={'seed':seed,'freq':freq,'optimizer':opt,'rule':rule,'steps':steps,'tau':tau,'seconds':elapsed,'accepted':accepted,'bc_uphill':uphill,'bc_violation_steps':violation,'max_bc':max(h['bc'] for h in hist),'candidate_evaluations':evals,'no_common_steps':no_common,'safe_better_steps':opportunity,**metrics(net,freq)}
    name=f'f{freq}_s{seed}_{opt}_{rule}'
    with (RAW/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=hist[0].keys());w.writeheader();w.writerows(hist)
    torch.save({'model':net.state_dict(),'optimizer_state':state,'summary':summary},RAW/f'{name}.pt')
    print(json.dumps(summary),flush=True)
    return summary

def main():
    global RAW
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='0,1,2');ap.add_argument('--freqs',default='1,3');ap.add_argument('--optimizers',default='adam,sgdm');ap.add_argument('--steps',type=int,default=200);ap.add_argument('--tau',type=float,default=1e-4);ap.add_argument('--tag',default='main');args=ap.parse_args()
    RAW = RAW / args.tag
    RAW.mkdir(parents=True,exist_ok=False)
    config={**vars(args),'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),'device':'cpu','dtype':'float64','threads':1,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'network':[1,24,24,1],'train_interior':64,'test_interior':511,'weights':[0,1,10,100],'lr_scales':[1,.25,.0625]}
    (RAW/f'config_{args.tag}.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    summaries=[]
    for freq in map(int,args.freqs.split(',')):
        for seed in map(int,args.seeds.split(',')):
            net,bc=prepare(seed,freq)
            torch.save({'model':net.state_dict(),'bc':bc},RAW/f'initial_f{freq}_s{seed}.pt')
            for opt in args.optimizers.split(','):
                for rule in ['strict','budget','free','vanilla']:
                    summaries.append(run(seed,freq,opt,rule,args.steps,args.tau,net))
                    (RAW/f'summary_{args.tag}.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    with (RAW/f'summary_{args.tag}.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=summaries[0].keys());w.writeheader();w.writerows(summaries)

if __name__=='__main__': main()
