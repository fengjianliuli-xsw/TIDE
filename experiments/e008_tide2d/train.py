import os,sys,time,json,copy,math,argparse,hashlib
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'e006_convergence'))
import converge as E
import torch,numpy as np
L=E.L;torch.set_num_threads(1)
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
METHODS=['linear_rank','gn_rank']
ANGLES=torch.arange(64)*2*math.pi/64
C=torch.stack((ANGLES.cos(),ANGLES.sin()),1)
def proposal(p,state):
    t=state['theta'];z=t.detach().requires_grad_();f=p.pde(z).square().sum();g=torch.autograd.grad(f,z)[0].detach()
    n=state['count']+1;m=.9*state['m']+.1*g;v=.999*state['v']+.001*g*g
    d=-.001*(m/(1-.9**n))/(torch.sqrt(v/(1-.999**n))+1e-8)
    return float(f.detach()),g,dict(theta=t,m=m,v=v,count=n,d=d)
def values(p,r,t,caps,pv):
    f,bs=p.values(t);rb=[float(y.square().sum()) for y in r.constraints(t)]
    safe=all(b<=c for b,c in zip(bs+rb,caps+caps))
    return dict(pde=f,small=bs,review=rb,safe=safe,pass_step=safe and f<pv)
def basis(p,r,t,d,caps,trial,rule='violation'):
    if rule=='violation':scores=[b/c-1 for b,c in zip(trial['small']+trial['review'],caps+caps)]
    else:
        current=[float(y.square().sum()) for pr in (p,r) for y in pr.constraints(t)]
        scores=[b/c-1 for b,c in zip(current,caps+caps)]
    i=int(np.argmax(scores));z=t.detach().requires_grad_();pr=(p,r)[i//2]
    loss=pr.constraints(z)[i%2].square().sum();w=-torch.autograd.grad(loss,z)[0].detach()
    dn=d.norm();u=d/dn if dn>0 else d;v=w-u*(w@u)
    if dn==0 or v.norm()<=1e-12*w.norm():return None,i
    return torch.stack((d,dn*v/v.norm()),1),i
def models(p,r,t,D):
    rows=[]
    for pr in (p,r):
        rr,j0=torch.func.jvp(pr.constraints,(t,),(D[:,0],));_,j1=torch.func.jvp(pr.constraints,(t,),(D[:,1],))
        for y,a,b in zip(rr,j0,j1):
            J=torch.stack((a,b),1);rows.append((float(y@y),2*(y@J),J.T@J))
    return rows
def ranks(blocks,caps,g,D):
    ls=torch.tensor([q[0] for q in blocks]);aa=torch.stack([q[1] for q in blocks]);GG=torch.stack([q[2] for q in blocks])
    ac=C@aa.T;qc=torch.einsum('ki,bij,kj->kb',C,GG,C).clamp_min(0);h=torch.tensor(caps+caps)-ls
    def bounds(q):
        hh=h.expand_as(ac);disc=(ac*ac+4*q*hh.clamp_min(0)).sqrt()
        stable=torch.where(ac>=0,2*hh/(disc+ac).clamp_min(1e-300),(disc-ac)/(2*q).clamp_min(1e-300))
        linear=torch.where(ac>0,hh/ac.clamp_min(1e-300),torch.ones_like(ac))
        return torch.where(hh<0,torch.zeros_like(ac),torch.where(q==0,linear,stable)).clamp(0,1).min(1).values
    al=bounds(torch.zeros_like(qc));ag=bounds(qc);gain=-(C@(g@D))
    return al,ag,gain
def select(p,r,state,caps,mode):
    start=time.perf_counter();pv,g,prop=proposal(p,state);t=state['theta'];d=prop['d'];checks=0;jvps=0;attempts=[]
    log=dict(trigger=False,block=None,angle_index=None,gram=None,jvp_seconds=0.,verify_seconds=0.)
    def verify(direction,alpha,halves):
        nonlocal checks
        st=time.perf_counter();v=values(p,r,t+alpha*direction,caps,pv);log['verify_seconds']+=time.perf_counter()-st;checks+=1
        attempts.append(dict(alpha=alpha,halves=halves,**v));return v
    full=verify(d,1.,0);log['full_proposal']=full
    direction=d;alpha=1.;first=1
    if full['pass_step']:
        accepted=True;chosen_halves=0
    else:
        accepted=False;chosen_halves=None
        if not full['safe']:
            log['trigger']=True;D,block=basis(p,r,t,d,caps,full);log['block']=block
            if D is not None:
                st=time.perf_counter();blocks=models(p,r,t,D);jvps=4;log['jvp_seconds']=time.perf_counter()-st
                al,ag,gain=ranks(blocks,caps,g,D);score=gain*(al if mode=='linear_rank' else ag);k=int(score.argmax())
                log.update(angle_index=k,linear_alpha=float(al[k]),gn_alpha=float(ag[k]),predicted_gain=float(score[k]),gram=[dict(loss=q[0],a=q[1].tolist(),G=q[2].tolist()) for q in blocks])
                if score[k]>0:direction=D@C[k];alpha=float(ag[k]);first=0
                else:alpha=0.
        for h in range(first,13) if alpha>0 and not accepted else []:
            vv=verify(direction,alpha*2**-h,h)
            if vv['pass_step']:accepted=True;chosen_halves=h;break
    actual_alpha=alpha*2**-chosen_halves if accepted else 0.
    log.update(attempts=attempts,jvps=jvps,checks=checks,alpha=actual_alpha,halves=chosen_halves,accepted=accepted,pde_before=pv,step_seconds=time.perf_counter()-start)
    new={k:prop[k] for k in ('m','v','count')};new['theta']=t+actual_alpha*direction
    return new if accepted else state,log
def run(name,seed,mode,hard):
    folder=OUT/f'{name}_s{seed}';folder.mkdir(exist_ok=True);path=folder/f'{mode}.pt'
    resume=None
    if path.exists():
        resume=torch.load(path,weights_only=False)
        if resume['stop']!='time_censored' or resume['train_seconds']>=hard:return
    ini=torch.load(L.OUT/f'{name}_s{seed}'/'initial.pt',weights_only=False);caps=ini['caps'];state=copy.deepcopy(ini['state'])
    p=L.Problem(name,seed);r=L.Problem(name,10000+seed,192,32);val=L.Problem(name,70000+seed,192,32)
    p0=val.values(state['theta'])[0];hist=[E.evaluate(val,state['theta'],caps,p0,0,0.)];snapshots=[dict(step=0,elapsed=0.,state=copy.deepcopy(state))]
    moves=[];step=0;hit=None;streak=0;nextcheck=2.;jvps=checks=triggers=direct=0;stop=None;plat=None;elapsed0=0.
    if resume:
        state=resume['state'];hist=resume['validation'];snapshots=resume['snapshots'];moves=resume['moves'];step=resume['steps'];hit=resume['target_confirmation'];elapsed0=resume['train_seconds'];nextcheck=2*(int(elapsed0/2)+1)
        jvps=resume['jvps'];checks=resume['checks'];triggers=resume['triggers'];direct=resume['direct_accepts']
        for h in reversed(hist):
            if not h['target']:break
            streak+=1
    st=time.perf_counter()-elapsed0
    with (folder/f'{mode}.jsonl').open('a' if resume else 'w',encoding='utf8') as file:
        while True:
            old=state['theta'];new,log=select(p,r,state,caps,mode);step+=1;elapsed=time.perf_counter()-st
            if log['accepted']:
                if torch.equal(old,new['theta']):stop='numerical_zero'
                else:state=new;moves.append((elapsed,float((state['theta']-old).norm()/max(1.,float(old.norm())))))
            else:stop='feasible_step_stagnation'
            jvps+=log['jvps'];checks+=log['checks'];triggers+=int(log['trigger']);direct+=int(log['full_proposal']['pass_step'])
            log.update(step=step,elapsed=elapsed);file.write(json.dumps(log)+'\n');file.flush()
            periodic=elapsed>=nextcheck
            if periodic or stop:
                snapshots.append(dict(step=step,elapsed=elapsed,state=copy.deepcopy(state)));hist.append(E.evaluate(val,state['theta'],caps,p0,step,time.perf_counter()-st))
                nextcheck=2*(int(hist[-1]['elapsed']/2)+1)
                if periodic:streak=streak+1 if hist[-1]['target'] else 0
                if streak>=3 and hit is None:
                    hit=dict(first=hist[-3],confirmation=hist[-1]);torch.save(copy.deepcopy(state),folder/f'{mode}_confirmed.pt')
                good,plat=E.platform(hist,moves,hist[-1]['elapsed'])
                if not stop and elapsed>=60 and good:stop='empirical_platform'
                if not stop and elapsed>=60 and plat and not plat['active']:stop='microstep_stagnation'
            if not stop and elapsed>=hard:stop='time_censored'
            if stop:break
    if hist[-1]['step']!=step:
        snapshots.append(dict(step=step,elapsed=time.perf_counter()-st,state=copy.deepcopy(state)));hist.append(E.evaluate(val,state['theta'],caps,p0,step,time.perf_counter()-st))
    result=dict(case=name,seed=seed,method=mode,initial=ini['state'],state=state,caps=caps,validation=hist,snapshots=snapshots,moves=moves,steps=step,accepted=state['count'],target_confirmation=hit,final_target=hist[-1]['target'],stop=stop,platform=plat,train_seconds=time.perf_counter()-st,hard_budget=hard,jvps=jvps,checks=checks,triggers=triggers,direct_accepts=direct,initial_validation_pde=p0,final_metrics=L.metrics(p,state['theta'],caps),continuation_origin_seconds=elapsed0)
    torch.save(result,path)
    if hard==120:torch.save(result,folder/f'{mode}_120.pt')
    L.dump(folder/f'{mode}_summary.json',{k:v for k,v in result.items() if k not in ['initial','state','validation','snapshots','moves']})
    print(name,seed,mode,hard,stop,step,round(result['train_seconds'],2),hist[-1]['pde_ratio'],flush=True)
def selftest():
    global values,models
    rows=[]
    for path in sorted(E.OUT.glob('*/tide_lazy.pt')):
        rec=torch.load(path,weights_only=False);p=L.Problem(rec['case'],rec['seed']);r=L.Problem(rec['case'],10000+rec['seed'],192,32);t=rec['state']['theta'];caps=rec['caps']
        pv,g,prop=proposal(p,rec['state']);trial=values(p,r,t+prop['d'],caps,pv);D,i=basis(p,r,t,prop['d'],caps,trial);Ds,j=basis(p,r,t,prop['d'],caps,trial,'slack')
        row=dict(case=rec['case'],seed=rec['seed'],max_violation_block=i,min_slack_block=j,same=i==j,full=trial)
        if D is not None:
            blocks=models(p,r,t,D);al,ag,gain=ranks(blocks,caps,g,D)
            for pr,off in ((p,0),(r,2)):
                _,mixed=torch.func.jvp(pr.constraints,(t,),(D@C[7],))
                for b,y in enumerate(mixed):torch.testing.assert_close(y@y,C[7]@blocks[off+b][2]@C[7],rtol=1e-9,atol=1e-15)
            for k in range(64):
                ref=min(L.R.bound(float(a@C[k]),max(0.,float(C[k]@G@C[k])),cap-loss) for (loss,a,G),cap in zip(blocks,caps+caps))
                assert abs(ref-float(ag[k]))<1e-10,(ref,ag[k])
            row.update(linear_angle=int((gain*al).argmax()),gn_angle=int((gain*ag).argmax()))
        rows.append(row)
    assert len(rows)==24
    L.dump(ROOT/'offline_same_snapshot.json',rows)
    # Decision branches through deterministic injected evaluations; no JVP on accepted full step or PDE-only rejection.
    original_values,original_models=values,models
    def forbidden(*args):raise AssertionError('unexpected JVP')
    models=forbidden
    ini=torch.load(L.OUT/'nls_s0'/'initial.pt',weights_only=False);p=L.Problem('nls',0);r=L.Problem('nls',10000,192,32)
    values=lambda p,r,t,caps,pv:dict(pde=pv*.9,small=[0.,0.],review=[0.,0.],safe=True,pass_step=True)
    _,log=select(p,r,ini['state'],ini['caps'],'gn_rank');assert log['jvps']==0 and log['checks']==1 and log['accepted']
    values=lambda p,r,t,caps,pv:dict(pde=pv*1.1,small=[0.,0.],review=[0.,0.],safe=True,pass_step=False)
    _,log=select(p,r,ini['state'],ini['caps'],'gn_rank');assert log['jvps']==0 and log['checks']==13 and not log['accepted'] and not log['trigger']
    values,models=original_values,original_models
    L.dump(ROOT/'selftest.json',dict(status='passed',snapshots=24,jvp_mixed_direction=True,vectorized_bounds=True,full_accept_no_jvp=True,pde_only_no_jvp=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    else:
        jobs=[(n,s,m) for n in L.CASES for s in range(5) for m in METHODS if (L.OUT/f'{n}_s{s}'/'initial.pt').exists()]
        for i,job in enumerate(jobs):
            if i%a.workers==a.worker:
                run(*job,120);run(*job,360)
