import os,time,math,json,csv,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplcache'))
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
CASES=['elasticity','kovasznay','nls','gray_scott','helmholtz']

def csvwrite(p,rows):
    if not rows:return
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def jet(theta,x,lo,hi,nout):
    nin=x.shape[1];z=2*(x-lo)/(hi-lo)-1
    ds=[torch.zeros_like(z) for _ in range(nin)]
    for k in range(nin):ds[k][:,k]=2/(hi[k]-lo[k])
    hs=[torch.zeros_like(z) for _ in range(nin)];off=0
    for layer,(a,b) in enumerate([(nin,24),(24,24),(24,nout)]):
        w=theta[off:off+a*b].view(b,a);off+=a*b;bias=theta[off:off+b];off+=b
        z=z@w.T+bias;ds=[v@w.T for v in ds];hs=[v@w.T for v in hs]
        if layer<2:
            z=torch.tanh(z);g=1-z*z
            hs=[g*h-2*z*g*d*d for h,d in zip(hs,ds)];ds=[g*d for d in ds]
    return z,ds,hs

def gray_ref(n,dt=.025,T=2.):
    q=np.arange(n)/n;x,y=np.meshgrid(q,q,indexing='ij');blob=(1+np.cos(2*np.pi*(x-.5)))*(1+np.cos(2*np.pi*(y-.5)))/4
    u=1-.3*blob;v=.15*blob
    k=2*np.pi*np.fft.fftfreq(n,d=1/n);lap=-(k[:,None]**2+k[None,:]**2)
    def rhs(u,v):
        uv=u*v*v
        return 2e-5*np.fft.ifft2(lap*np.fft.fft2(u)).real-uv+.04*(1-u),1e-5*np.fft.ifft2(lap*np.fft.fft2(v)).real+uv-.10*v
    snapshots=[np.stack([u,v],-1)]
    steps=round(T/dt)
    for i in range(steps):
        a,b=rhs(u,v);c,d=rhs(u+dt*a/2,v+dt*b/2);e,f=rhs(u+dt*c/2,v+dt*d/2);g,h=rhs(u+dt*e,v+dt*f)
        u=u+dt*(a+2*c+2*e+g)/6;v=v+dt*(b+2*d+2*f+h)/6
        if (i+1)%(steps//4)==0:snapshots.append(np.stack([u,v],-1))
    return np.stack(snapshots)

class Problem:
    def __init__(self,name,seed=0,n=48,nb=16):
        self.name=name;self.nout={'elasticity':5,'kovasznay':3,'nls':2,'gray_scott':2,'helmholtz':1}[name]
        self.dim=3 if name=='gray_scott' else 2
        self.lo=torch.tensor([-5.,0.]) if name=='nls' else torch.zeros(self.dim)
        self.hi=torch.tensor([5.,2.]) if name=='nls' else torch.ones(self.dim)
        if name=='gray_scott':self.hi[-1]=2.
        gen=torch.Generator().manual_seed(902+seed);self.x=self.lo+(self.hi-self.lo)*torch.rand(n,self.dim,generator=gen)
        self.names=['bc','ic'] if name in ['nls','gray_scott'] else ['bc']
        self.edges=[]
        spatial=1 if name=='nls' else 2
        for k in range(spatial):
            a=self.lo+(self.hi-self.lo)*torch.rand(nb,self.dim,generator=gen);b=a.clone();a[:,k]=self.lo[k];b[:,k]=self.hi[k];self.edges.append((a,b))
        self.xb=torch.cat([p for pair in self.edges for p in pair])
        self.xic=self.lo+(self.hi-self.lo)*torch.rand(nb*2,self.dim,generator=gen);self.xic[:,-1]=0.
        if name=='gray_scott':
            # Value and normal derivative periodic constraints; IC remains separate.
            self.nb=nb
    def forward(self,t,x):return jet(t,x,self.lo,self.hi,self.nout)
    def exact(self,x):
        a,b=x[:,0],x[:,1];p=math.pi
        if self.name=='elasticity':
            u=torch.sin(p*a)*torch.sin(p*b);v=.5*torch.sin(2*p*a)*torch.sin(p*b)
            ux=p*torch.cos(p*a)*torch.sin(p*b);vy=.5*p*torch.sin(2*p*a)*torch.cos(p*b)
            uy=p*torch.sin(p*a)*torch.cos(p*b);vx=p*torch.cos(2*p*a)*torch.sin(p*b)
            mu=1/2.6;lam=.3/(1.3*.4)
            return torch.stack([u,v,(lam+2*mu)*ux+lam*vy,lam*ux+(lam+2*mu)*vy,mu*(uy+vx)],1)
        if self.name=='kovasznay':
            lam=20-math.sqrt(400+4*p*p);e=torch.exp(lam*a)
            return torch.stack([1-e*torch.cos(2*p*b),lam/(2*p)*e*torch.sin(2*p*b),.5*(1-e*e)],1)
        if self.name=='nls':
            amp=1/torch.cosh(a);return torch.stack([amp*torch.cos(.5*b),amp*torch.sin(.5*b)],1)
        if self.name=='helmholtz':return (torch.sin(p*a)*torch.sin(2*p*b)+.1*torch.cos(p*a)*torch.cos(p*b))[:,None]
        blob=(1+torch.cos(2*p*(a-.5)))*(1+torch.cos(2*p*(b-.5)))/4
        return torch.stack([1-.3*blob,.15*blob],1)
    def constraints(self,t):
        if self.name=='gray_scott':
            parts=[]
            for k,(a,b) in enumerate(self.edges):
                za,da,_=self.forward(t,a);zb,db,_=self.forward(t,b)
                parts +=[(za-zb).flatten(),(da[k]-db[k]).flatten()]
            bc=torch.cat(parts)
        else:
            y=self.forward(t,self.xb)[0];target=self.exact(self.xb)
            if self.name=='elasticity':bc=(y[:,:2]-target[:,:2]).flatten()
            elif self.name=='kovasznay':bc=torch.cat([(y[:,:2]-target[:,:2]).flatten(),(y[:1,2]-target[:1,2])])
            else:bc=(y-target).flatten()
        out=[bc]
        if len(self.names)>1:out.append((self.forward(t,self.xic)[0]-self.exact(self.xic)).flatten())
        return tuple(r/math.sqrt(r.numel()) for r in out)
    def pde(self,t,x=None):
        x=self.x if x is None else x;y,d,h=self.forward(t,x);a,b=x[:,0],x[:,1];p=math.pi
        if self.name=='elasticity':
            mu=1/2.6;lam=.3/(1.3*.4);ux,vy=d[0][:,0],d[1][:,1]
            # Analytic divergence of exact isotropic stress (plane strain E=1,nu=.3).
            uxx=-p*p*torch.sin(p*a)*torch.sin(p*b);uyy=uxx
            vxy=p*p*torch.cos(2*p*a)*torch.cos(p*b)
            uxy=p*p*torch.cos(p*a)*torch.cos(p*b)
            vxx=-2*p*p*torch.sin(2*p*a)*torch.sin(p*b);vyy=-.5*p*p*torch.sin(2*p*a)*torch.sin(p*b)
            divx=(lam+2*mu)*uxx+mu*uyy+(lam+mu)*vxy
            divy=mu*vxx+(lam+2*mu)*vyy+(lam+mu)*uxy
            parts=[d[0][:,2]+d[1][:,4]-divx,d[0][:,4]+d[1][:,3]-divy,y[:,2]-(lam+2*mu)*ux-lam*vy,y[:,3]-lam*ux-(lam+2*mu)*vy,y[:,4]-mu*(d[1][:,0]+d[0][:,1])]
        elif self.name=='kovasznay':
            u,v=y[:,0],y[:,1];nu=.025
            parts=[u*d[0][:,0]+v*d[1][:,0]+d[0][:,2]-nu*(h[0][:,0]+h[1][:,0]),u*d[0][:,1]+v*d[1][:,1]+d[1][:,2]-nu*(h[0][:,1]+h[1][:,1]),d[0][:,0]+d[1][:,1]]
        elif self.name=='nls':
            amp=y.square().sum(1);parts=[-d[1][:,1]+.5*h[0][:,0]+amp*y[:,0],d[1][:,0]+.5*h[0][:,1]+amp*y[:,1]]
        elif self.name=='gray_scott':
            u,v=y[:,0],y[:,1];uv=u*v*v
            parts=[d[2][:,0]-2e-5*(h[0][:,0]+h[1][:,0])+uv-.04*(1-u),d[2][:,1]-1e-5*(h[0][:,1]+h[1][:,1])-uv+.10*v]
        else:
            exact=self.exact(x)[:,0];s=torch.sin(p*a)*torch.sin(2*p*b);c=.1*torch.cos(p*a)*torch.cos(p*b)
            forcing=(25-5*p*p)*s+(25-2*p*p)*c
            parts=[h[0][:,0]+h[1][:,0]+25*y[:,0]-forcing]
        return torch.stack(parts,1).flatten()/math.sqrt(len(parts)*len(x))
    def values(self,t):return float(self.pde(t).square().sum()),[float(r.square().sum()) for r in self.constraints(t)]
    def metrics(self,t):
        test=Problem(self.name,991,n=384,nb=48);rp=float(test.pde(t).square().sum());cs=[float(r.square().sum()) for r in test.constraints(t)]
        if self.name=='gray_scott':
            ref=np.load(ROOT/'gray_reference.npz')['fine'];n=ref.shape[1];pts=[];truth=[]
            for j,timeval in enumerate(np.linspace(0,2,5)):
                q=torch.arange(n)/n;xx,yy=torch.meshgrid(q,q,indexing='ij');pts.append(torch.stack([xx.flatten(),yy.flatten(),torch.full_like(xx.flatten(),float(timeval))],1));truth.append(torch.from_numpy(ref[j].reshape(-1,2)))
            x=torch.cat(pts);truth=torch.cat(truth)
        else:x=test.x;truth=self.exact(x)
        pred=self.forward(t,x)[0]
        return {'test_pde':rp,'test_constraints':cs,'relative_l2':float((pred-truth).norm()/truth.norm()),'component_relative_l2':[float((pred[:,j]-truth[:,j]).norm()/(truth[:,j].norm()+1e-14)) for j in range(self.nout)]}

def init(problem,seed):
    torch.manual_seed(seed);dims=[problem.dim,24,24,problem.nout];parts=[]
    for a,b in zip(dims[:-1],dims[1:]):
        layer=nn.Linear(a,b);nn.init.xavier_normal_(layer.weight);nn.init.zeros_(layer.bias);parts.extend([layer.weight.detach().flatten(),layer.bias.detach()])
    t=torch.cat(parts).requires_grad_();opt=torch.optim.Adam([t],lr=.003)
    for i in range(1200):
        opt.zero_grad();loss=sum(r.square().sum() for r in problem.constraints(t));loss.backward();opt.step()
        if loss.item()<1e-5:break
    return t.detach().clone(),i+1

def grads(problem,t):
    z=t.detach().requires_grad_();p=problem.pde(z).square().sum();rs=problem.constraints(z)
    gp=torch.autograd.grad(p,z,retain_graph=True)[0].detach();gs=[];bs=[]
    for r in rs:
        l=r.square().sum();bs.append(float(l.detach()));gs.append(torch.autograd.grad(l,z,retain_graph=True)[0].detach())
    return float(p.detach()),bs,gp,gs

def train(problem,t0,method,seed,steps,out,budget=None):
    t=t0.clone();m=torch.zeros_like(t);v=torch.zeros_like(t);count=0
    _,b0,_,_=grads(problem,t);caps=[max(1e-4,2*b) for b in b0]
    checks=jvps=accepted=violations=false_safe=0;history=[];start=time.perf_counter();constraint_seconds=jvp_seconds=pde_seconds=0.
    for step in range(steps):
        if budget and step>0 and time.perf_counter()-start>=budget:break
        pv,bs,gp,gs=grads(problem,t);gb=sum(gs);candidates=[]
        for weight in [0.,1.,100.]:
            g=gp+weight*gb;mt=.9*m+.1*g;vt=.999*v+.001*g*g;ct=count+1
            d=-.001*(mt/(1-.9**ct))/(torch.sqrt(vt/(1-.999**ct))+1e-8)
            a=[float(gi@d) for gi in gs];q=None
            candidates.append({'d':d,'m':mt,'v':vt,'count':ct,'a':a,'q':q,'name':weight,'score':float(gp@d)})
        candidates.sort(key=lambda c:c['score']);chosen=None
        if method=='always':
            for c in candidates:
                st=time.perf_counter();r,jd=torch.func.jvp(problem.constraints,(t,),(c['d'],));c['q']=[float(x.square().sum()) for x in jd];jvps+=1;jvp_seconds+=time.perf_counter()-st
        for c in candidates:
            alpha=1.
            for attempt in range(8):
                if c['q'] is not None:
                    for base,cap,a,q in zip(bs,caps,c['a'],c['q']):
                        margin=max(0.,cap-base)
                        if q>1e-30:
                            root=(-a+math.sqrt(a*a+4*q*margin))/(2*q)
                            alpha=min(alpha,root*.99)
                        elif a>0:alpha=min(alpha,.99*margin/a)
                if alpha<1e-9:break
                z=t+alpha*c['d'];st=time.perf_counter();cb=[float(r.square().sum()) for r in problem.constraints(z)];checks+=1;constraint_seconds+=time.perf_counter()-st
                safe=all(b<=cap for b,cap in zip(cb,caps))
                if not safe:
                    false_safe+=1
                    if method=='lazy' and c['q'] is None:
                        st=time.perf_counter();r,jd=torch.func.jvp(problem.constraints,(t,),(c['d'],));c['q']=[float(x.square().sum()) for x in jd];jvps+=1;jvp_seconds+=time.perf_counter()-st
                    else:alpha*=.5
                    continue
                st=time.perf_counter();pp=float(problem.pde(z).square().sum());pde_seconds+=time.perf_counter()-st
                if pp<pv:chosen=(c,alpha,z,pp,cb);break
                alpha*=.5
            if chosen:break
        if chosen:
            c,alpha,t,pp,cb=chosen;m,v,count=c['m'],c['v'],c['count'];accepted+=1
        else:alpha=0.;pp,cb=pv,bs
        violations+=int(any(b>cap for b,cap in zip(cb,caps)))
        history.append({'step':step+1,'pde':pp,'constraints':json.dumps(cb),'alpha':alpha,'accepted':int(chosen is not None),'checks':checks,'jvps':jvps,'elapsed':time.perf_counter()-start})
    seconds=time.perf_counter()-start
    result={'case':problem.name,'seed':seed,'method':method,'steps':len(history),'accepted':accepted,'seconds':seconds,'checks':checks,'jvps':jvps,'constraint_seconds':constraint_seconds,'jvp_seconds':jvp_seconds,'pde_trial_seconds':pde_seconds,'violations':violations,'unsafe_trials':false_safe,'initial_constraints':b0,'caps':caps,**problem.metrics(t)}
    name=f'{problem.name}_s{seed}_{method}';csvwrite(out/f'{name}.csv',history);torch.save({'theta':t,'m':m,'v':v,'count':count},out/f'{name}.pt');print(json.dumps(result),flush=True);return result

def validate():
    rows=[]
    for name in CASES:
        problem=Problem(name,n=8,nb=4);torch.manual_seed(0);t,_=init(problem,0)
        x=problem.x;z,ds,hs=problem.forward(t,x);eps=1e-4
        for k in range(problem.dim):
            e=torch.zeros_like(x);e[:,k]=eps
            zp=problem.forward(t,x+e)[0];zm=problem.forward(t,x-e)[0]
            assert float(((zp-zm)/(2*eps)-ds[k]).abs().max())<1e-5
            assert float(((zp-2*z+zm)/(eps**2)-hs[k]).abs().max())<1e-4
        # Verify analytic fields satisfy each PDE independently of the network jets.
        if name!='gray_scott':
            xx=x.detach().requires_grad_();y=problem.exact(xx)
            d=[];h=[]
            for k in range(problem.dim):
                dd=[];hh=[]
                for j in range(problem.nout):
                    first=torch.autograd.grad(y[:,j].sum(),xx,create_graph=True,retain_graph=True)[0][:,k]
                    second=torch.autograd.grad(first.sum(),xx,create_graph=True,retain_graph=True)[0][:,k]
                    dd.append(first);hh.append(second)
                d.append(torch.stack(dd,1));h.append(torch.stack(hh,1))
            old=problem.forward;problem.forward=lambda t,x:(y,d,h)
            err=float(problem.pde(t,xx).abs().max());problem.forward=old
            assert err<1e-10,(name,err)
        else:err=None
        direction=torch.randn_like(t)*1e-4;rs,jv=torch.func.jvp(problem.constraints,(t,),(direction,))
        a=[float(2*(r*v).sum()) for r,v in zip(rs,jv)]
        _,_,_,gs=grads(problem,t)
        assert max(abs(v-float(g@direction)) for v,g in zip(a,gs))<1e-10
        rows.append({'case':name,'analytic_residual_max':err,'jet_finite_difference':'passed','constraint_jvp':'passed'})
    (ROOT/'validation.json').write_text(json.dumps(rows,indent=2));print('All five equation/jet/JVP validations passed.',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--steps',type=int,default=120);ap.add_argument('--seeds',default='0,1,2');ap.add_argument('--cases',default=','.join(CASES));ap.add_argument('--budget',type=float);args=ap.parse_args()
    out=ROOT/args.tag;out.mkdir(exist_ok=False);config={**vars(args),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'torch':torch.__version__,'precision':'float64','device':'cpu','width':24,'n_pde':48,'boundary_points_per_edge':16,'pretrain_steps_max':1200,'caps':'max(1e-4,2*initial_block_MSE)'}
    (out/'config.json').write_text(json.dumps(config,indent=2));(out/'source.py').write_bytes(Path(__file__).read_bytes())
    if not (ROOT/'gray_reference.npz').exists():
        coarse=gray_ref(32,.05);fine=gray_ref(64,.025)
        comparison=np.linalg.norm(coarse-fine[:,::2,::2])/np.linalg.norm(fine[:,::2,::2]);np.savez_compressed(ROOT/'gray_reference.npz',coarse=coarse,fine=fine)
        (ROOT/'reference_check.json').write_text(json.dumps({'relative_32_dt05_vs64_dt025':float(comparison),'T':2.,'snapshots':5,'reference':'periodic Fourier Laplacian + RK4; smooth IC'},indent=2))
    validate();results=[]
    for name in args.cases.split(','):
        for seed in map(int,args.seeds.split(',')):
            problem=Problem(name,seed);t,presteps=init(problem,seed);torch.save({'theta':t,'pretrain_steps':presteps},out/f'{name}_s{seed}_initial.pt')
            for method in ['fo','always','lazy']:
                results.append(train(problem,t,method,seed,args.steps,out,args.budget));(out/'summary.json').write_text(json.dumps(results,indent=2))
    csvwrite(out/'summary.csv',results)

if __name__=='__main__':main()
