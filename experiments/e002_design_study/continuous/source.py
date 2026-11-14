import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplcache'))
import argparse,csv,json,math,time,hashlib,sys
import numpy as np
import torch
sys.path.insert(0,str(ROOT.parent))
from run_experiment import Net,vec,proposal
torch.set_num_threads(1);torch.set_default_dtype(torch.float64)

def savecsv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

class Problem:
    def __init__(self,freq,n=48):
        self.freq=freq;self.x=torch.linspace(0,1,n+2)[1:-1,None].requires_grad_();self.xb=torch.tensor([[0.],[1.]])
    def forward(self,t,x):
        off=0
        for ni,no in [(1,24),(24,24),(24,1)]:
            w=t[off:off+ni*no].view(no,ni);off+=ni*no
            b=t[off:off+no];off+=no
            x=x@w.T+b
            if no!=1:x=torch.tanh(x)
        return x
    def bc(self,t):return (self.forward(t,self.xb)-self.xb).flatten()
    def residual(self,t):
        u=self.forward(t,self.x)
        ux=torch.autograd.grad(u.sum(),self.x,create_graph=True)[0]
        uxx=torch.autograd.grad(ux.sum(),self.x,create_graph=True)[0]
        rp=uxx+(self.freq*math.pi)**2*torch.sin(self.freq*math.pi*self.x)
        return rp.flatten()/math.sqrt(self.x.numel()),self.bc(t)/math.sqrt(2)
    def values(self,t):
        p,b=self.residual(t);return float(p.square().sum().detach()),float(b.square().sum().detach())
    def derivatives(self,t):
        q=t.detach().requires_grad_();rp,rb=self.residual(q);p,b=rp.square().sum(),rb.square().sum()
        gp=torch.autograd.grad(p,q,retain_graph=True)[0].detach();gb=torch.autograd.grad(b,q,retain_graph=True)[0].detach()
        jb=torch.stack([torch.autograd.grad(rb[i],q,retain_graph=True)[0] for i in range(2)]).detach()
        return float(p.detach()),float(b.detach()),gp,gb,jb,rb.detach()
    def jvp(self,t,d):
        r,v=torch.autograd.functional.jvp(self.residual,t,d,create_graph=False)
        a=[float(2*(x*y).sum()) for x,y in zip(r,v)]
        b=[float(y.square().sum()) for y in v]
        return a,b
    def metrics(self,t):
        test=Problem(self.freq,511);p,b=test.values(t)
        x=torch.linspace(0,1,513)[:,None];truth=torch.sin(self.freq*math.pi*x)+x
        rel=float((self.forward(t,x)-truth).norm()/truth.norm())
        return {'test_pde':p,'bc':b,'relative_l2':rel}

def initialize(seed,mode):
    torch.manual_seed(seed);t=vec(Net()).requires_grad_();problem=Problem(1)
    if mode=='prefit':
        opt=torch.optim.Adam([t],lr=.003)
        for _ in range(800):
            opt.zero_grad();b=problem.bc(t).square().mean();b.backward();opt.step()
            if b.item()<1e-10:break
    return t.detach().clone()

def candidates(t,state,gp,gb,jb,opt,bank):
    lr=.001 if opt=='adam' else .0001
    out=[]
    for weight in [0.,1.,100.,1e4]:
        d,ns=proposal(gp+weight*gb,state,opt,lr);out.append((f'w{weight:g}',d,ns))
    if bank=='tangent':
        d,ns=out[0][1:]
        projected=d-jb.T@torch.linalg.solve(jb@jb.T+1e-12*torch.eye(2),jb@d)
        ns={k:(v.clone() if torch.is_tensor(v) else v) for k,v in ns.items()}
        if opt=='adam':
            ns['m']=(-projected/lr)*(torch.sqrt(ns['v']/(1-.999**ns['t']))+1e-8)*(1-.9**ns['t'])
        else:ns['m']=-projected/lr
        out.append(('tangent',projected,ns))
    return out

METHODS={
 'oracle_raw':('oracle','raw','budget',False),
 'fo_raw':('fo','raw','budget',False),
 'gn_raw':('gn','raw','budget',False),
 'oracle_tangent':('oracle','tangent','budget',False),
 'fo_tangent':('fo','tangent','budget',False),
 'gn_tangent':('gn','tangent','budget',False),
 'gn_strict_tangent':('gn','tangent','strict',False),
 'gn_repair_raw':('gn','raw','budget',True),
 'fo_all_raw':('fo','raw','budget',False),
 'fo_all_tangent':('fo','tangent','budget',False),
 'fo_strict_tangent':('fo','tangent','strict',False),
 'gn_cont_tangent':('gn','tangent','budget',False),
 'pair_oracle_raw':('pair','raw','budget',False),
}

def train(seed,freq,opt,method,steps,out,tau=1e-4):
    pred,bank,rule,repair=METHODS[method];problem=Problem(freq)
    t=initialize(seed,'prefit');initial=t.clone()
    state={'t':0,'m':torch.zeros_like(t),'v':torch.zeros_like(t)}
    hist=[];audit=[];checks=0;jvps=0;false_safe=0;accepted=0;repairs=0;start=time.perf_counter()
    for step in range(steps):
        pv,bv,gp,gb,jb,rb=problem.derivatives(t);cap=bv+1e-12 if rule=='strict' else tau
        cs=candidates(t,state,gp,gb,jb,opt,bank);rank=[]
        for name,d,ns in cs:
            if pred=='pair':
                for alpha in [1.,.25,.0625,.015625]:
                    z=t+alpha*d
                    jz=torch.autograd.functional.jacobian(problem.bc,z)/math.sqrt(2)
                    rz=problem.bc(z).detach()/math.sqrt(2)
                    correction=-jz.T@torch.linalg.solve(jz@jz.T+1e-12*torch.eye(2),rz)
                    total=alpha*d+correction
                    pp,bb=problem.values(t+total);checks+=1
                    reset={'t':0,'m':torch.zeros_like(t),'v':torch.zeros_like(t)}
                    if pp<pv and bb<=cap:rank.append((pp,bb,'pair_'+name,total,reset,1.))
                continue
            if pred=='gn':a,b=problem.jvp(t,d);jvps+=1
            elif pred=='fo':a=[float(gp@d),float(gb@d)];b=[0.,0.]
            alphas=[1.,.25,.0625,.015625]
            if method=='gn_cont_tangent':
                margin=max(0.,cap-bv)
                if b[1]>1e-30:
                    safe=2*margin/(a[1]+math.sqrt(a[1]**2+4*b[1]*margin)) if a[1]>=0 and margin>0 else (-a[1]+math.sqrt(a[1]**2+4*b[1]*margin))/(2*b[1])
                else:safe=margin/a[1] if a[1]>0 else 1.
                optimum=-a[0]/(2*b[0]) if b[0]>1e-30 else 1.
                best=max(0.,min(1.,safe,optimum))
                alphas=[best,best*.5,best*.25,best*.0625]
            for alpha in alphas:
                if pred=='oracle':pp,bb=problem.values(t+alpha*d);checks+=1
                else:pp=pv+alpha*a[0]+alpha*alpha*b[0];bb=bv+alpha*a[1]+alpha*alpha*b[1]
                if np.isfinite(pp+bb) and pp<pv and bb<=cap:
                    rank.append((pp,bb,name,d,ns,alpha))
        rank.sort(key=lambda c:c[0]);selected=None
        for c in (rank if method.startswith('fo_all') else rank[:3]):
            pp,bb=problem.values(t+c[5]*c[3]);checks+=1
            if bb>cap:false_safe+=1
            if bb<=cap and pp<pv:
                selected=(c,pp,bb);break
        mode='reject';name='none';alpha=0.;pp,bb=pv,bv
        if selected:
            c,pp,bb=selected;_,_,name,d,state,alpha=c;t=t+alpha*d;mode='progress';accepted+=1
        elif repair and bv>1e-12:
            # Minimum-norm residual repair, capped to stay within base proposal scale.
            d=-jb.T@torch.linalg.solve(jb@jb.T+1e-12*torch.eye(2),rb)
            for alpha in [1.,.25,.0625,.015625]:
                pr,br=problem.values(t+alpha*d);checks+=1
                if br<.5*bv and br<=tau:
                    t=t+alpha*d;pp,bb=pr,br;mode='repair';name='normal';repairs+=1;accepted+=1
                    # A repair is outside the base optimizer; reset momentum explicitly.
                    state={'t':0,'m':torch.zeros_like(t),'v':torch.zeros_like(t)};break
        hist.append({'step':step+1,'pde':pp,'bc':bb,'mode':mode,'direction':name,'alpha':alpha,'delta_pde':pp-pv,'delta_bc':bb-bv,'checks_total':checks,'jvps_total':jvps})
        if step in [0,steps//2,steps-1]:
            torch.save({'theta':t,'state':state},out/f'f{freq}_s{seed}_{opt}_{method}_step{step+1}.pt')
    seconds=time.perf_counter()-start
    name=f'f{freq}_s{seed}_{opt}_{method}'
    savecsv(out/f'{name}.csv',hist)
    summary={'seed':seed,'freq':freq,'optimizer':opt,'method':method,'steps':steps,'seconds':seconds,'accepted':accepted,'repairs':repairs,'bc_uphill':sum(h['delta_bc']>1e-12 for h in hist),'violations':sum(h['bc']>tau+1e-14 for h in hist),'max_bc':max(h['bc'] for h in hist),'checks':checks,'jvps':jvps,'false_safe':false_safe,**problem.metrics(t)}
    print(json.dumps(summary),flush=True);return summary

def replay(out):
    rows=[]
    for seed in [0,1,2]:
        for freq in [1,3]:
            problem=Problem(freq);t=initialize(seed,'random').requires_grad_();opt=torch.optim.Adam([t],lr=.001)
            for step in range(301):
                if step in [0,30,100,300]:
                    z=t.detach().clone();pv,bv,gp,gb,jb,rb=problem.derivatives(z)
                    # Preserve actual baseline Adam state at each checkpoint.
                    st=opt.state.get(t,{})
                    state={'t':int(st.get('step',0)),'m':st.get('exp_avg',torch.zeros_like(z)).clone(),'v':st.get('exp_avg_sq',torch.zeros_like(z)).clone()}
                    for radius in [.001,.01,.05]:
                        cs=candidates(z,state,gp,gb,jb,'adam','tangent');curves=[]
                        for name,d,_ in cs:
                            d=d/(d.norm()+1e-30)*radius;a,b=problem.jvp(z,d);ap,ab=problem.values(z+d)
                            curves.append((name,a,b,ap,ab))
                        mean_curv=float(np.mean([c[2][1] for c in curves]));cap=1e-4 if bv<=1e-4 else bv
                        for idx,(name,a,b,ap,ab) in enumerate(curves):
                            shuffled=curves[(idx+1)%len(curves)][2][1]
                            for estimator,cp,cb in [('fo',0.,0.),('gn',b[0],b[1]),('isotropic',float(np.mean([c[2][0] for c in curves])),mean_curv),('shuffled',curves[(idx+1)%len(curves)][2][0],shuffled)]:
                                predp,predb=pv+a[0]+cp,bv+a[1]+cb
                                rows.append({'seed':seed,'freq':freq,'checkpoint':step,'radius':radius,'direction':name,'estimator':estimator,'base_pde':pv,'base_bc':bv,'cap':cap,'actual_pde':ap,'actual_bc':ab,'pred_pde':predp,'pred_bc':predb,'bc_error':abs(predb-ab),'false_safe':int(predb<=cap and ab>cap),'false_reject':int(predb>cap and ab<=cap)})
                if step<300:
                    opt.zero_grad();rp,rb=problem.residual(t);l=rp.square().sum()+100*rb.square().sum();l.backward();opt.step()
    savecsv(out/'replay_candidates.csv',rows)
    decisions=[]
    keys=sorted(set((r['seed'],r['freq'],r['checkpoint'],r['radius']) for r in rows))
    for key in keys:
        rs=[r for r in rows if (r['seed'],r['freq'],r['checkpoint'],r['radius'])==key]
        truth=[r for r in rs if r['estimator']=='fo' and r['actual_bc']<=r['cap'] and r['actual_pde']<r['base_pde']]
        oracle=min(truth,key=lambda r:r['actual_pde']) if truth else None
        for estimator in ['fo','gn','isotropic','shuffled']:
            avail=[r for r in rs if r['estimator']==estimator and r['pred_bc']<=r['cap'] and r['pred_pde']<r['base_pde']]
            pick=min(avail,key=lambda r:r['pred_pde']) if avail else None
            feasible=bool(pick and pick['actual_bc']<=pick['cap'] and pick['actual_pde']<pick['base_pde'])
            decisions.append({'seed':key[0],'freq':key[1],'checkpoint':key[2],'radius':key[3],'estimator':estimator,'oracle_exists':int(oracle is not None),'picked':pick['direction'] if pick else 'none','feasible_pick':int(feasible),'missed_opportunity':int(oracle is not None and not feasible),'regret':pick['actual_pde']-oracle['actual_pde'] if feasible and oracle else ''})
    savecsv(out/'replay_decisions.csv',decisions)

def validate():
    p=Problem(1);t=initialize(0,'prefit');pv,bv,gp,gb,jb,rb=p.derivatives(t)
    d=torch.randn_like(t);d=d/d.norm()*.001;a,b=p.jvp(t,d)
    assert abs(a[0]-float(gp@d))<1e-10
    assert abs(a[1]-float(gb@d))<1e-10
    assert abs(b[1]-float((jb@d).square().sum()))<1e-14
    print('JVP linear terms match parameter gradients; boundary quadratic matches full residual Jacobian.',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--steps',type=int,default=150);ap.add_argument('--seeds',default='0,1,2');ap.add_argument('--freqs',default='1,3');ap.add_argument('--optimizer',default='adam');ap.add_argument('--methods',default=','.join(METHODS));ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    out=ROOT/args.tag;out.mkdir(exist_ok=False)
    conf={**vars(args),'tau':1e-4,'device':'cpu','dtype':'float64','torch':torch.__version__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'config.json').write_text(json.dumps(conf,indent=2));(out/'source.py').write_bytes(Path(__file__).read_bytes());validate()
    if args.replay:replay(out)
    results=[]
    for seed in map(int,args.seeds.split(',')):
        for freq in map(int,args.freqs.split(',')):
            for method in args.methods.split(','):
                if not method:continue
                results.append(train(seed,freq,args.optimizer,method,args.steps,out))
                (out/'summary.json').write_text(json.dumps(results,indent=2))
    savecsv(out/'summary.csv',results)

if __name__=='__main__':main()
