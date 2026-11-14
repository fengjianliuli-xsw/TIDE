"""Frozen E005 continuous experiment. CPU float64, one thread, fixed points."""
import os,sys,time,json,math,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
sys.path.insert(0,str(ROOT.parent/'e003_on_demand_five'))
sys.path.insert(0,str(ROOT.parent/'e004_tide_score_replay'))
import bench as B
import replay as R
import numpy as np
import torch
from scipy.io import loadmat
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.interpolate import RegularGridInterpolator
torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
OUT=ROOT/'results_long';DATA=ROOT/'data';DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
CASES=['nls','gray_scott','cgle','fhn','allen_cahn']
METHODS=['adam','slope_gn','tide_full','tide_lazy']
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def spectral_cgle(n,dt):
    x=np.arange(n)*2*np.pi/n;a=.6+.25*np.exp(1j*x)+.1*np.exp(-2j*x)
    k=np.fft.fftfreq(n,1/n);L=1-(1+1j)*k*k
    # Integrating-factor RK4 avoids explicit diffusion stiffness; cubic evaluated in x.
    E=np.exp(L*dt/2);E2=E*E;v=np.fft.fft(a);sn=[a.copy()];steps=round(2/dt)
    def N(v):z=np.fft.ifft(v);return np.fft.fft(-(1+.5j)*abs(z)**2*z)
    for j in range(steps):
        aa=dt*N(v);bb=dt*N(E*(v+aa/2));cc=dt*N(E*v+bb/2);dd=dt*N(E2*v+E*cc)
        v=E2*v+(E2*aa+2*E*(bb+cc)+dd)/6
        if (j+1)%(steps//20)==0:sn.append(np.fft.ifft(v))
    return np.stack(sn)
def fhn_reference(n=50,tight=False):
    # Literal PDEBench sim_diff_react Simulator algorithm and parameters, truncated T=5.
    dx=2/n;rng=np.random.default_rng(0);z=np.r_[rng.standard_normal(n*n),rng.standard_normal(n*n)]
    main=-4*np.ones((n,n))/dx**2;main[:,[0,-1]]+=1/dx**2;main[[0,-1],:]+=1/dx**2
    left=np.ones(n);left[0]=0;left=np.tile(left,n)[1:]/dx**2
    right=np.ones(n);right[-1]=0;right=np.tile(right,n)[:-1]/dx**2
    lap=diags([main.ravel(),left,right,np.ones(n*(n-1))/dx**2,np.ones(n*(n-1))/dx**2],[0,-1,1,-n,n])
    def fun(t,z):u,v=np.split(z,2);return np.r_[.001*(lap@u)+u-u**3-.005-v,.005*(lap@v)+u-v]
    tt=np.linspace(0,5,51);kw={'rtol':1e-8,'atol':1e-10} if tight else {}
    sol=solve_ivp(fun,(0,5),z,t_eval=tt,**kw);assert sol.success
    return sol.y.T.reshape(51,2,n,n).transpose(0,2,3,1)
def prepare_data():
    start=time.perf_counter();meta={}
    p=DATA/'cgle.npz'
    if not p.exists():
        c=spectral_cgle(64,.002);f=spectral_cgle(128,.001)
        np.savez(p,coarse=c,fine=f,x=np.arange(128)*2*np.pi/128,t=np.linspace(0,2,21))
    d=np.load(p);meta['cgle']={'coarse_fine_relative':float(np.linalg.norm(d['coarse']-d['fine'][:,::2])/np.linalg.norm(d['fine'][:,::2]))}
    p=DATA/'fhn.npz'
    if not p.exists():
        c=fhn_reference();f=fhn_reference(tight=True);q=-1+(np.arange(50)+.5)*.04
        np.savez(p,coarse=c,fine=f,x=q,y=q,t=np.linspace(0,5,51))
    d=np.load(p);meta['fhn']={'coarse_fine_relative':float(np.linalg.norm(d['coarse']-d['fine'])/np.linalg.norm(d['fine'])),'variant':'PDEBench generator, default seed=0, 50x50, white-noise IC; T=5 instead of50; default solve_ivp versus tight same-grid check; no spatial refinement'}
    d=np.load(ROOT.parent/'e003_on_demand_five'/'gray_reference.npz');meta['gray_scott']={k:list(d[k].shape) for k in d.files}
    meta['gray_scott']['coarse_fine_relative']=float(np.linalg.norm(d['coarse']-d['fine'][:,::2,::2])/np.linalg.norm(d['fine'][:,::2,::2]))
    if (DATA/'fhn_first.h5').exists():
        import h5py
        op=DATA/'fhn_official.npz'
        if not op.exists():
            with h5py.File(DATA/'fhn_first.h5') as f:
                g=f['0000'];np.savez(op,fine=g['data'][:].astype(np.float64),x=g['grid/x'][:].astype(np.float64),y=g['grid/y'][:].astype(np.float64),t=g['grid/t'][:].astype(np.float64))
        meta['fhn_main']={'source':'official PDEBench 2D diffusion-reaction trajectory 0000, data/fhn_first.h5','grid':[101,128,128,2],'source_config':'512x512, 501 time points, T5, seed0; distributed data downsampled','IC':'fixed bilinear interpolation, clip coordinates to available cell centers; no noise extrapolation','refinement':'official single distributed grid only; generated50 grid comparison does not validate official trajectory'}
    meta['seconds']=time.perf_counter()-start;dump(DATA/'reference_audit.json',meta)

class Problem(B.Problem):
    def __init__(self,name,seed=0,n=48,nb=16):
        if name in ['nls','gray_scott']:super().__init__(name,seed,n,nb);return
        self.name=name;self.dim=3 if name=='fhn' else 2;self.nout=1 if name=='allen_cahn' else 2
        self.lo=torch.tensor([-1.,-1.,0.]) if name=='fhn' else torch.tensor([-1.,0.] if name=='allen_cahn' else [0.,0.])
        self.hi=torch.tensor([1.,1.,5.]) if name=='fhn' else torch.tensor([1.,1.] if name=='allen_cahn' else [2*math.pi,2.])
        gen=torch.Generator().manual_seed(902+seed);self.x=self.lo+(self.hi-self.lo)*torch.rand(n,self.dim,generator=gen)
        self.names=['bc','ic'];self.edges=[]
        for k in range(self.dim-1):
            a=self.lo+(self.hi-self.lo)*torch.rand(nb,self.dim,generator=gen);b=a.clone();a[:,k]=self.lo[k];b[:,k]=self.hi[k]
            a[:,-1]=torch.clamp(a[:,-1],min=1e-6);b[:,-1]=a[:,-1];self.edges.append((a,b))
        self.xic=self.lo+(self.hi-self.lo)*torch.rand(2*nb,self.dim,generator=gen);self.xic[:,-1]=0
        if name=='fhn':
            d=np.load(DATA/'fhn_official.npz');self.ic_bounds=(float(d['y'][0]),float(d['y'][-1]),float(d['x'][0]),float(d['x'][-1]));self.ic=RegularGridInterpolator((d['y'],d['x']),d['fine'][0],bounds_error=True)
            self.ic_target=self.exact(self.xic)
    def exact(self,x):
        if self.name in ['nls','gray_scott']:return super().exact(x)
        if self.name=='cgle':
            a=x[:,0];return torch.stack([.6+.25*torch.cos(a)+.1*torch.cos(2*a),.25*torch.sin(a)-.1*torch.sin(2*a)],1)
        if self.name=='allen_cahn':return (x[:,0]**2*torch.cos(math.pi*x[:,0]))[:,None]
        pts=x[:,:2].detach().numpy()[:,::-1].copy();ya,yb,xa,xb=self.ic_bounds;pts[:,0]=np.clip(pts[:,0],ya,yb);pts[:,1]=np.clip(pts[:,1],xa,xb)
        return torch.from_numpy(self.ic(pts))
    def constraints(self,t):
        if self.name in ['nls','gray_scott']:return super().constraints(t)
        parts=[]
        for k,(a,b) in enumerate(self.edges):
            za,da,_=self.forward(t,a);zb,db,_=self.forward(t,b)
            parts.extend([da[k].flatten(),db[k].flatten()] if self.name=='fhn' else [(za-zb).flatten(),(da[k]-db[k]).flatten()])
        target=self.ic_target if self.name=='fhn' else self.exact(self.xic)
        rr=[torch.cat(parts),(self.forward(t,self.xic)[0]-target).flatten()]
        return tuple(r/math.sqrt(r.numel()) for r in rr)
    def pde(self,t,x=None):
        if self.name in ['nls','gray_scott']:return super().pde(t,x)
        x=self.x if x is None else x;y,d,h=self.forward(t,x)
        if self.name=='cgle':
            u,v=y.T;s=u*u+v*v;parts=[d[1][:,0]-u-h[0][:,0]+h[0][:,1]+s*(u-.5*v),d[1][:,1]-v-h[0][:,1]-h[0][:,0]+s*(v+.5*u)]
        elif self.name=='fhn':
            u,v=y.T;parts=[d[2][:,0]-.001*(h[0][:,0]+h[1][:,0])-u+u**3+.005+v,d[2][:,1]-.005*(h[0][:,1]+h[1][:,1])-u+v]
        else:parts=[d[1][:,0]-1e-4*h[0][:,0]-5*(y[:,0]-y[:,0]**3)]
        return torch.stack(parts,1).flatten()/math.sqrt(len(parts)*len(x))

def eval_grid(name):
    if name=='nls':
        xx,tt=np.meshgrid(np.linspace(-5,5,128),np.linspace(0,2,21),indexing='ij');x=np.stack([xx.ravel(),tt.ravel()],1);a=1/np.cosh(xx)*np.exp(.5j*tt);truth=np.stack([a.real.ravel(),a.imag.ravel()],1)
    elif name=='gray_scott':
        d=np.load(ROOT.parent/'e003_on_demand_five'/'gray_reference.npz');truth=d['fine'];n=truth.shape[1];tt,xx,yy=np.meshgrid(np.linspace(0,2,len(truth)),np.arange(n)/n,np.arange(n)/n,indexing='ij');x=np.stack([xx.ravel(),yy.ravel(),tt.ravel()],1);truth=truth.reshape(-1,2)
    elif name=='cgle':
        d=np.load(DATA/'cgle.npz');tt,xx=np.meshgrid(d['t'],d['x'],indexing='ij');x=np.stack([xx.ravel(),tt.ravel()],1);a=d['fine'].ravel();truth=np.stack([a.real,a.imag],1)
    elif name=='fhn':
        d=np.load(DATA/'fhn_official.npz');tt,yy,xx=np.meshgrid(d['t'][::5],d['y'],d['x'],indexing='ij');x=np.stack([xx.ravel(),yy.ravel(),tt.ravel()],1);truth=d['fine'][::5].reshape(-1,2)
    else:
        d=loadmat(DATA/'AC.mat');xx=d['x'].ravel();tt=d['tt'].ravel();xx2,tt2=np.meshgrid(xx,tt,indexing='ij');x=np.stack([xx2.ravel(),tt2.ravel()],1);truth=d['uu'].real.reshape(-1,1)
    return torch.tensor(x),torch.tensor(truth)
def metrics(prob,t,caps):
    test=Problem(prob.name,991,384,48);pv,bs=test.values(t);x,y=eval_grid(prob.name)
    with torch.no_grad():pred=torch.cat([prob.forward(t,q)[0] for q in x.split(4096)])
    r={'relative_l2':float((pred-y).norm()/y.norm()),'components':[float((pred[:,j]-y[:,j]).norm()/y[:,j].norm()) for j in range(prob.nout)],'audit_pde':pv,'audit_constraints':bs,'audit_excess':[max(0.,b/c-1) for b,c in zip(bs,caps)],'audit_violations':[b>c for b,c in zip(bs,caps)]}
    if prob.name in ['nls','cgle']:r['amplitude_relative_l2']=float((pred.norm(dim=1)-y.norm(dim=1)).norm()/y.norm(dim=1).norm())
    return r
def initial(prob,review,seed):
    torch.manual_seed(seed);parts=[]
    for a,b in zip([prob.dim,24,24],[24,24,prob.nout]):
        w=torch.empty(b,a);torch.nn.init.xavier_normal_(w);parts.extend([w.flatten(),torch.zeros(b)])
    t=torch.cat(parts).requires_grad_();opt=torch.optim.Adam([t],lr=.003);start=time.perf_counter()
    for k in range(12000):
        opt.zero_grad();loss=sum(r.square().sum() for p in [prob,review] for r in p.constraints(t));loss.backward();opt.step()
        if float(loss.detach())<1e-5:break
        if k>=1199 and (k+1)%200==0:
            with torch.no_grad():
                sb=[float(r.square().sum()) for r in prob.constraints(t)];rb=[float(r.square().sum()) for r in review.constraints(t)]
            if all(b<=max(1e-4,2*s) for s,b in zip(sb,rb)):break
    t=t.detach();_,bs=prob.values(t);caps=[max(1e-4,2*b) for b in bs];_,rb=review.values(t)
    assert all(b<=c for b,c in zip(rb,caps)),(prob.name,seed,bs,rb,caps)
    return {'theta':t,'m':torch.zeros_like(t),'v':torch.zeros_like(t),'count':0},caps,{'steps':k+1,'seconds':time.perf_counter()-start,'small':bs,'review':rb}
def context(prob,state,caps):
    pv,bs,gp,gs=B.grads(prob,state['theta']);count=state['count']+1;cs=[];h=[cap-b for cap,b in zip(caps,bs)]
    for cid,w in enumerate([0.,1.,100.]):
        g=gp+w*sum(gs);mt=.9*state['m']+.1*g;vt=.999*state['v']+.001*g*g
        d=-.001*(mt/(1-.9**count))/(torch.sqrt(vt/(1-.999**count))+1e-8)
        c={'id':cid,'weight':w,'m':mt,'v':vt,'count':count,'d':d,'p':float(-gp@d),'a':[float(gg@d) for gg in gs]}
        c['linear_alpha']=R.alpha(c,[0.]*len(bs),h);c['upper']=c['p']*c['linear_alpha'];cs.append(c)
    return {'theta':state['theta'],'problem':prob,'pv':pv,'bs':bs,'h':h,'caps':caps,'cs':cs}
def adam_context(prob,state):
    z=state['theta'].detach().requires_grad_();p=prob.pde(z).square().sum();rs=prob.constraints(z);bs=[r.square().sum() for r in rs]
    g=torch.autograd.grad(p+100*sum(bs),z)[0].detach();count=state['count']+1;mt=.9*state['m']+.1*g;vt=.999*state['v']+.001*g*g
    d=-.001*(mt/(1-.9**count))/(torch.sqrt(vt/(1-.999**count))+1e-8)
    return {'pv':float(p.detach()),'bs':[float(b.detach()) for b in bs]}, {'id':2,'d':d,'m':mt,'v':vt,'count':count}
def choose(ctx,review,method):
    cache={};jvps=checks=0;qt=vt=0.;attempts=[];candidates=ctx['cs'];remaining=[c for c in candidates if c['p']>0]
    def score(c):
        nonlocal jvps,qt
        if c['id'] not in cache:
            st=time.perf_counter();q=R.qcalc(ctx,c);qt+=time.perf_counter()-st;jvps+=1;al=R.alpha(c,q,ctx['h']);cache[c['id']]={'q':q,'alpha':al,'score':c['p']*al,'limiting_block':int(np.argmin([R.bound(a,b,h) for a,b,h in zip(c['a'],q,ctx['h'])]))}
        return cache[c['id']]
    if method=='tide_full':
        for c in remaining:score(c)
    chosen=None;aa=0.
    while remaining:
        if method in ['slope_gn','direct']:c=min(remaining,key=lambda x:(-x['p'],x['id']))
        elif method=='linear':c=min(remaining,key=lambda x:(-x['upper'],x['id']))
        elif method=='tide_full':c=min(remaining,key=lambda x:(-score(x)['score'],x['id']))
        else:
            best=None
            for cc in sorted(remaining,key=lambda x:(-x['upper'],x['id'])):
                if best is not None and cc['upper']<best[0]-1e-14*max(1,abs(best[0])):break
                s=score(cc)
                if best is None or (-s['score'],cc['id'])<(-best[0],best[1]):best=(s['score'],cc['id'])
            c=next(x for x in remaining if x['id']==best[1])
        al=c['linear_alpha'] if method=='linear' else (1. if method=='direct' else score(c)['alpha'])
        # Direct-forward ablation uses a fixed 8-point dyadic schedule, charges every trial.
        schedule=[al/2**i for i in range(8)] if method=='direct' else [al]
        for a in schedule:
            st=time.perf_counter();r=R.actual(ctx,c,a);rb=[float(q.square().sum()) for q in review.constraints(ctx['theta']+a*c['d'])];safe=all(b<=cap for b,cap in zip(rb,ctx['caps']));vt+=time.perf_counter()-st;checks+=1
            reason='accept' if r['pass'] and safe else ('review' if not safe else ('small' if not r['safe'] else 'pde_or_zero'))
            attempts.append({'id':c['id'],'alpha':a,'pde':r['pde'],'small':r['constraints'],'review':rb,'reason':reason})
            if r['pass'] and safe:chosen=c;aa=a;break
        if chosen is not None:break
        remaining=[x for x in remaining if x['id']!=c['id']]
    rows=[{'id':c['id'],'p':c['p'],'a':c['a'],'upper':c['upper'],'linear_alpha':c['linear_alpha'],**cache.get(c['id'],{})} for c in candidates]
    return chosen,aa,{'candidates':rows,'attempts':attempts,'jvps':jvps,'checks':checks,'jvp_seconds':qt,'verify_seconds':vt,'final_id':-1 if chosen is None else chosen['id'],'alpha':aa}
def run(prob,review,initial,caps,method,folder,budget=None,maxsteps=1000):
    path=folder/(method+'.pt')
    if path.exists():return torch.load(path,weights_only=False)
    state={k:v.clone() if torch.is_tensor(v) else v for k,v in initial.items()};history=[];thetas=[state['theta'].clone()];elapsed=[0.];start=time.perf_counter();stop='step_limit';totalj=0
    with (folder/(method+'.jsonl')).open('w',encoding='utf-8') as f:
        for k in range(maxsteps):
            # Retain a 1000-update synchronous trajectory, then slice at measured budget.
            # Budget-excluded suffix is never used for the equal-time comparison.
            if method=='adam':
                ctx,c=adam_context(prob,state);a=1.;log={'final_id':2,'alpha':1.,'jvps':0,'checks':0,'candidates':[],'attempts':[]}
            else:ctx=context(prob,state,caps);c,a,log=choose(ctx,review,method)
            if c is not None:state={'theta':state['theta']+a*c['d'],'m':c['m'],'v':c['v'],'count':c['count']}
            tm=time.perf_counter()-start;log.update(step=k+1,accepted=c is not None,elapsed=tm,pde_before=ctx['pv'],small_before=ctx['bs']);totalj+=log['jvps'];f.write(json.dumps(log)+'\n');f.flush();history.append(log)
            if c is None:stop='stagnation';break
            thetas.append(state['theta'].clone());elapsed.append(tm)
    wall=time.perf_counter()-start
    # Every committed theta retained, permits exact budget and common-prefix reconstruction.
    result={'initial':initial,'state':state,'caps':caps,'thetas':torch.stack(thetas),'elapsed':elapsed,'history':history,'stop':stop,'train_seconds':wall,'budget':budget,'accepted':len(thetas)-1,'jvps':totalj}
    if budget is None:budget=wall
    ix=max(i for i,t in enumerate(elapsed) if t<=budget);result['budget_index']=ix;result['budget_metrics']=metrics(prob,thetas[ix],caps);result['final_metrics']=metrics(prob,state['theta'],caps);result['initial_metrics']=metrics(prob,initial['theta'],caps)
    torch.save(result,path);return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cases',nargs='+',default=CASES);ap.add_argument('--seeds',nargs='+',type=int,default=list(range(5)));ap.add_argument('--steps',type=int,default=1000);args=ap.parse_args()
    prepare_data();dump(OUT/'config.json',{'cases':args.cases,'seeds':args.seeds,'steps':args.steps,'weights':[0,1,100],'lr':.001,'cap':'max(1e-4,2*small_initial_block_MSE)','points':'train48 small16 review192/32 audit384/48','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'dtype':'float64','threads':1,'budget':'paired per-seed 1000-step weighted Adam measured training wall; methods run through max(budget,1000 updates) for sync; no beyond-budget checkpoint in main comparison'})
    summary=[]
    for name in args.cases:
        for seed in args.seeds:
            folder=OUT/f'{name}_s{seed}';folder.mkdir(exist_ok=True);prob=Problem(name,seed);review=Problem(name,10000+seed,192,32);ip=folder/'initial.pt'
            if ip.exists():ini=torch.load(ip,weights_only=False);state,caps=ini['state'],ini['caps']
            else:
                if (folder/'startup_failure.json').exists():print('STARTUP_FAILED_CACHED',name,seed,flush=True);continue
                ps=time.perf_counter()
                try:state,caps,prep=initial(prob,review,seed)
                except AssertionError as e:
                    dump(folder/'startup_failure.json',{'case':name,'seed':seed,'steps':12000,'seconds':time.perf_counter()-ps,'detail':e.args,'status':'small-review joint initialization did not reach E004 cap; no main comparison; no cap relaxation'});print('STARTUP_FAILED',name,seed,e,flush=True);continue
                ini={'state':state,'caps':caps,'preparation':prep};torch.save(ini,ip)
            # Exclude one warmup of each path, including torch.func.jvp dispatch initialization.
            warm=time.perf_counter();adam_context(prob,state);choose(context(prob,state,caps),review,'tide_full');warm=time.perf_counter()-warm
            print('START',name,seed,flush=True);adam=run(prob,review,state,caps,'adam',folder,maxsteps=args.steps);budget=adam['train_seconds']
            for method in METHODS:
                r=adam if method=='adam' else run(prob,review,state,caps,method,folder,budget,maxsteps=args.steps)
                row={'case':name,'seed':seed,'method':method,'accepted':r['accepted'],'budget_accepted':r['budget_index'],'stop':r['stop'],'budget_seconds':budget,'train_seconds':r['train_seconds'],'preparation_seconds':ini['preparation']['seconds'],'jvps':r['jvps'],**r['budget_metrics']};summary.append(row);print('DONE',name,seed,method,r['accepted'],r['stop'],r['budget_metrics']['relative_l2'],flush=True);dump(OUT/'summary.json',summary)
            # Cost and pure-linear ablations on shared initial and terminal Adam snapshots only.
            abl=[]
            for sn,ss in [('initial',state),('adam_terminal',adam['state'])]:
                ctx=context(prob,ss,caps)
                rb=[float(q.square().sum()) for q in review.constraints(ss['theta'])]
                if any(h<0 for h in ctx['h']) or any(b>cap for b,cap in zip(rb,caps)):abl.append({'snapshot':sn,'eligible':False,'small':ctx['bs'],'review':rb});continue
                for m in ['tide_full','tide_lazy','linear','direct']:
                    st=time.perf_counter();_,_,log=choose(ctx,review,m);abl.append({'snapshot':sn,'method':m,'eligible':True,'selection_seconds':time.perf_counter()-st,**log})
            dump(folder/'ablations.json',abl)
    dump(OUT/'completion.json',{'time':time.strftime('%Y-%m-%d %H:%M:%S'),'runs':len(summary)})
if __name__=='__main__':main()
