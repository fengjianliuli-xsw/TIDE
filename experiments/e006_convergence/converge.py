import os,sys,time,json,copy,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
sys.path.insert(0,str(ROOT.parent/'e005_evolution_suite'))
import longrun as L
import torch,numpy as np
torch.set_num_threads(1)
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
METHODS=['adam','slope_gn','tide_lazy']
CONFIG={'validation_seed':70000,'validation_n':192,'validation_nb':32,'check_seconds':2,'window_seconds':10,'window_checks':4,'platform_tolerance':.01,'min_seconds':60,'hard_seconds':120,'minimum_effective_updates':20,'window_relative_displacement':1e-8,'target_ratio':.1,'confirm_checks':3,'shrink':[2.**-i for i in range(13)],'threads':1,'concurrency':3}

def jet_tangent(t,v,x,lo,hi,no):
    nin=x.shape[1];z=2*(x-lo)/(hi-lo)-1;tz=torch.zeros_like(z)
    ds=[torch.zeros_like(z) for _ in range(nin)]
    for k in range(nin):ds[k][:,k]=2/(hi[k]-lo[k])
    hs=[torch.zeros_like(z) for _ in ds];td=[torch.zeros_like(z) for _ in ds];th=[torch.zeros_like(z) for _ in ds];off=0
    for layer,(a,b) in enumerate([(nin,24),(24,24),(24,no)]):
        w=t[off:off+a*b].view(b,a);tw=v[off:off+a*b].view(b,a);off+=a*b
        bias=t[off:off+b];tb=v[off:off+b];off+=b
        tz=tz@w.T+z@tw.T+tb;z=z@w.T+bias
        td=[u@w.T+d@tw.T for u,d in zip(td,ds)];ds=[d@w.T for d in ds]
        th=[u@w.T+h@tw.T for u,h in zip(th,hs)];hs=[h@w.T for h in hs]
        if layer<2:
            z=torch.tanh(z);g=1-z*z;tz=g*tz;tg=-2*z*tz
            th=[tg*h+g*u-2*(tz*g+z*tg)*d*d-4*z*g*d*q for h,u,d,q in zip(hs,th,ds,td)]
            hs=[g*h-2*z*g*d*d for h,d in zip(hs,ds)]
            td=[tg*d+g*q for d,q in zip(ds,td)];ds=[g*d for d in ds]
    return tz,td,th

def qcalc(ctx,c):
    p=ctx['problem'];q=copy.copy(p)
    q.forward=lambda t,x:jet_tangent(ctx['theta'],c['d'],x,p.lo,p.hi,p.nout)
    q.exact=lambda x:torch.zeros((len(x),p.nout),dtype=x.dtype)
    if hasattr(p,'ic_target'):q.ic_target=torch.zeros_like(p.ic_target)
    return [float(r.square().sum()) for r in q.constraints(ctx['theta'])]

def select(ctx,review,method):
    cache={};attempts=[];jtime=0.;vtime=0.;checks=0
    viable=[c for c in ctx['cs'] if c['p']>0]
    def score(c):
        nonlocal jtime
        if c['id'] not in cache:
            st=time.perf_counter();q=qcalc(ctx,c);jtime+=time.perf_counter()-st
            a=L.R.alpha(c,q,ctx['h']);cache[c['id']]={'q':q,'alpha':a,'score':a*c['p']}
        return cache[c['id']]
    if method=='tide_full':
        for c in viable:score(c)
    # At each common shrink factor all original positive-step candidates compete anew.
    for shrink in CONFIG['shrink']:
        remain=list(viable)
        while remain:
            if method=='slope_gn':c=min(remain,key=lambda c:(-c['p'],c['id']))
            elif method=='tide_full':c=min(remain,key=lambda c:(-score(c)['score'],c['id']))
            else:
                best=None
                for cc in sorted(remain,key=lambda c:(-c['upper'],c['id'])):
                    if best is not None and cc['upper']<best[0]-1e-14*max(1,abs(best[0])):break
                    ss=score(cc)
                    if best is None or (-ss['score'],cc['id'])<(-best[0],best[1]):best=(ss['score'],cc['id'])
                c=next(cc for cc in remain if cc['id']==best[1])
            a=score(c)['alpha']*shrink
            if a>0:
                st=time.perf_counter();r=L.R.actual(ctx,c,a);rb=[float(z.square().sum()) for z in review.constraints(ctx['theta']+a*c['d'])];vtime+=time.perf_counter()-st;checks+=1
                ok=r['pass'] and all(b<=cap for b,cap in zip(rb,ctx['caps']))
                attempts.append({'id':c['id'],'alpha':a,'shrink':shrink,'pde':r['pde'],'small':r['constraints'],'review':rb,'pass':ok})
                if ok:return c,a,{'attempts':attempts,'q':cache,'jvps':len(cache),'checks':checks,'jvp_seconds':jtime,'verify_seconds':vtime,'shrink':shrink}
            remain=[cc for cc in remain if cc['id']!=c['id']]
    return None,0.,{'attempts':attempts,'q':cache,'jvps':len(cache),'checks':checks,'jvp_seconds':jtime,'verify_seconds':vtime,'shrink':0}

def evaluate(p,t,caps,p0,step,elapsed):
    pv,bs=p.values(t)
    return {'step':step,'elapsed':elapsed,'pde':pv,'pde_ratio':pv/p0,'constraints':bs,'cap_ratios':[b/c for b,c in zip(bs,caps)],'target':pv/p0<=.1 and all(b<=c for b,c in zip(bs,caps))}

def platform(hist,moves,now):
    groups=[[h for h in hist if now-(j+1)*10<h['elapsed']<=now-j*10] for j in [2,1,0]]
    if any(len(g)<4 for g in groups):return False,None
    med=np.array([[np.median([h['pde_ratio'] for h in g])]+[np.median([h['cap_ratios'][i] for h in g]) for i in range(2)] for g in groups])
    # All three metrics use fixed initial PDE / cap scales, including window spreads.
    improv=med[:-1]-med[1:]
    spans=np.array([[np.quantile([h['pde_ratio'] for h in g],.9)-np.quantile([h['pde_ratio'] for h in g],.1)]+[np.quantile([h['cap_ratios'][i] for h in g],.9)-np.quantile([h['cap_ratios'][i] for h in g],.1) for i in range(2)] for g in groups])
    updates=[m for m in moves if m[0]>now-30]
    active=len(updates)>=20 and sum(m[1] for m in updates)>=1e-8
    return bool(active and np.all(abs(improv)<=.01) and np.all(spans<=.02)),{'quantile_spans':spans.tolist(),'window_bounds':[[now-30,now-20],[now-20,now-10],[now-10,now]],'medians':med.tolist(),'relative_changes':improv.tolist(),'effective_updates':len(updates),'relative_displacement':sum(m[1] for m in updates),'active':active}

def run(name,seed,method,hard=120):
    folder=OUT/f'{name}_s{seed}';folder.mkdir(exist_ok=True);path=folder/f'{method}.pt'
    resume=None
    if path.exists():
        if hard<=120:return
        resume=torch.load(path,weights_only=False)
        if resume['stop']!='time_censored' or resume['train_seconds']>=hard:return
        archive=folder/f'{method}_120.pt'
        if not archive.exists():torch.save(resume,archive)
    old=L.OUT/f'{name}_s{seed}'
    if not (old/'initial.pt').exists():return
    ini=torch.load(old/'initial.pt',weights_only=False);state=copy.deepcopy(ini['state']);caps=ini['caps']
    prob=L.Problem(name,seed);review=L.Problem(name,10000+seed,192,32);val=L.Problem(name,70000+seed,192,32)
    snapshots=[{'step':0,'state':copy.deepcopy(state)}];p0=val.values(state['theta'])[0];hist=[evaluate(val,state['theta'],caps,p0,0,0.)];moves=[];st=time.perf_counter();nextcheck=2.;step=0;hit=None;streak=0;stop=None;plat=None;jvps=checks=0;zeros=0
    if resume is not None:
        state=resume['state'];hist=resume['validation'];snapshots=resume['snapshots'];moves=resume.get('moves',[]);step=resume['steps'];hit=resume['target_confirmation'];jvps=resume['jvps'];checks=resume['checks'];zeros=resume['numerical_zeros']
        streak=0
        for h in reversed(hist):
            if not h['target']:break
            streak+=1
        platform_after=resume['train_seconds']+30 if 'moves' not in resume else 60
        st=time.perf_counter()-resume['train_seconds'];nextcheck=2*(int(resume['train_seconds']/2)+1)
    with (folder/f'{method}.jsonl').open('a' if resume else 'w',encoding='utf8') as f:
        while True:
            step+=1
            if method=='adam':ctx,c=L.adam_context(prob,state);a=1.;log={'jvps':0,'checks':0,'attempts':[]}
            else:ctx=L.context(prob,state,caps);c,a,log=select(ctx,review,method)
            elapsed=time.perf_counter()-st
            if c is not None:
                delta=a*c['d'];rel=float(delta.norm()/max(1.,float(state['theta'].norm())))
                if torch.equal(state['theta']+delta,state['theta']):zeros+=1;stop='numerical_zero'
                else:state={'theta':state['theta']+delta,'m':c['m'],'v':c['v'],'count':c['count']};moves.append((elapsed,rel))
            else:stop='feasible_step_stagnation'
            jvps+=log['jvps'];checks+=log['checks'];log['candidates']=[{k:v for k,v in cc.items() if k not in ['d','m','v']} for cc in ctx.get('cs',[])]
            log.update(step=step,elapsed=elapsed,accepted=c is not None,alpha=a,final_id=-1 if c is None else c['id'],pde_before=ctx['pv'],small_before=ctx['bs'])
            f.write(json.dumps(log)+'\n');f.flush()
            periodic=elapsed>=nextcheck
            if periodic or stop:
                snapshots.append({'step':step,'state':copy.deepcopy(state)});hist.append(evaluate(val,state['theta'],caps,p0,step,time.perf_counter()-st));nextcheck=2*(int(hist[-1]['elapsed']/2)+1)
                streak=(streak+1 if hist[-1]['target'] else 0) if periodic else streak
                if streak>=3 and hit is None:
                    hit={'confirmation':hist[-1],'first':hist[-3]};torch.save(copy.deepcopy(state),folder/f'{method}_confirmed.pt')
                good,plat=platform(hist,moves,hist[-1]['elapsed'])
                (folder/f'{method}_validation.json').write_text(json.dumps(hist),encoding='utf8')
                if stop is None and elapsed>=max(60,platform_after if resume else 60) and good:stop='empirical_platform'
                if stop is None and elapsed>=max(60,platform_after if resume else 60) and plat and not plat['active']:stop='microstep_stagnation'
            if stop is None and elapsed>=hard:stop='time_censored'
            if stop:break
    if hist[-1]['step']!=step:hist.append(evaluate(val,state['theta'],caps,p0,step,time.perf_counter()-st))
    result={'continuation_origin_seconds':resume['train_seconds'] if resume else 0,'stage1_config':resume['config'] if resume else CONFIG,'hard_budget':hard,'moves':moves,'snapshots':snapshots,'case':name,'seed':seed,'method':method,'initial':ini['state'],'state':state,'caps':caps,'validation':hist,'target_confirmation':hit,'final_target':hist[-1]['target'],'stop':stop,'platform':plat,'accepted':state['count'],'steps':step,'train_seconds':time.perf_counter()-st,'unused_budget':max(0,hard-(time.perf_counter()-st)),'numerical_zeros':zeros,'jvps':jvps,'checks':checks,'final_metrics':L.metrics(prob,state['theta'],caps),'initial_validation_pde':p0,'config':CONFIG}
    torch.save(result,path)
    L.dump(folder/f'{method}_summary.json',{k:v for k,v in result.items() if k not in ['initial','state','validation','snapshots','moves']})
    print(name,seed,method,stop,step,result['train_seconds'],hist[-1]['pde_ratio'],flush=True)

def selftest():
    rows=[]
    for name in L.CASES:
        p=L.Problem(name,0);ini=torch.load(L.OUT/f'{name}_s0'/'initial.pt',weights_only=False);ctx=L.context(p,ini['state'],ini['caps']);c=ctx['cs'][0]
        q=qcalc(ctx,c);_,j=torch.func.jvp(p.constraints,(ctx['theta'],),(c['d'],));ref=[float(r.square().sum()) for r in j]
        err=max(abs(a-b)/max(1e-30,abs(b)) for a,b in zip(q,ref));assert err<1e-9,(name,err)
        rows.append({'case':name,'analytic':q,'torch':ref,'max_relative_error':err})
    L.dump(ROOT/'jvp_crosscheck.json',rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--cases',nargs='+',default=L.CASES);ap.add_argument('--selftest',action='store_true');ap.add_argument('--hard',type=int,default=120);a=ap.parse_args()
    if a.selftest:selftest()
    else:
        if a.hard>120:CONFIG['concurrency']=a.workers
        if a.hard>120:L.dump(ROOT/'extension_config.json',{**CONFIG,'hard_seconds':a.hard,'platform_after_resume_seconds':30,'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        elif not (ROOT/'frozen_config.json').exists():L.dump(ROOT/'frozen_config.json',{**CONFIG,'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        jobs=[(n,s,m) for n in a.cases for s in range(5) for m in METHODS if (L.OUT/f'{n}_s{s}'/'initial.pt').exists()]
        if a.hard>120:
            manifest=ROOT/'extension_jobs.json'
            if manifest.exists():jobs=[tuple(j) for j in json.loads(manifest.read_text(encoding='utf8'))]
            else:raise RuntimeError('Create immutable extension_jobs.json before starting workers')
        for i,job in enumerate(jobs):
            if (i if a.hard>120 else i//3)%a.workers==a.worker:run(*job,hard=a.hard)
