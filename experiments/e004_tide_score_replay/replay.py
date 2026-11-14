import os, sys, time, math, json, csv, hashlib, platform
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'e003_on_demand_five'
sys.path.insert(0,str(BASE))
os.environ['OMP_NUM_THREADS']='1'
import bench as B
import torch
import numpy as np
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
OUT=ROOT/'results'
METHODS=['full_gain','tide_full','tide_lazy','oracle']
def dump(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def write(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def bound(a,q,h):
    # Feasibility at alpha=0 is required. No artificial inward safety factor.
    if h<0:return 0.
    if q==0:return min(1.,h/a) if a>0 else 1.
    if h==0:return min(1.,-a/q) if a<0 else 0.
    disc=math.sqrt(a*a+4*q*h)
    root=2*h/(disc+a) if a>=0 else (disc-a)/(2*q)
    return max(0.,min(1.,root))
def alpha(c,q,h):return min(bound(a,b,hh) for a,b,hh in zip(c['a'],q,h))
def prepare(problem,state,caps,mult):
    t=state['theta'];pv,bs,gp,gs=B.grads(problem,t)
    m=state.get('m',torch.zeros_like(t));v=state.get('v',torch.zeros_like(t));count=state.get('count',0)+1
    cs=[]
    for cid,weight in enumerate([0.,1.,100.]):
        g=gp+weight*sum(gs);mt=.9*m+.1*g;vt=.999*v+.001*g*g
        d=mult*(-.001*(mt/(1-.9**count))/(torch.sqrt(vt/(1-.999**count))+1e-8))
        c={'id':cid,'weight':weight,'d':d,'p':float(-gp@d),'a':[float(gb@d) for gb in gs]}
        c['linear_alpha']=alpha(c,[0.]*len(bs),[cap-b for cap,b in zip(caps,bs)])
        c['upper']=c['p']*c['linear_alpha'];cs.append(c)
    return {'problem':problem,'theta':t,'pv':pv,'bs':bs,'caps':caps,'h':[cap-b for cap,b in zip(caps,bs)],'cs':cs}
def qcalc(ctx,c):
    _,j=torch.func.jvp(ctx['problem'].constraints,(ctx['theta'],),(c['d'],))
    return [float(x.square().sum()) for x in j]
def actual(ctx,c,al):
    z=ctx['theta']+al*c['d'];pp,bs=ctx['problem'].values(z)
    # Exact comparison; at the boundary the quadratic model can fail legitimately.
    safe=all(b<=cap for b,cap in zip(bs,ctx['caps']))
    return {'pde':pp,'constraints':bs,'gain':ctx['pv']-pp,'safe':safe,'pass':safe and pp<ctx['pv'] and al>0}
def select(ctx,method):
    start=time.perf_counter();cache={};jvps=checks=0;reject=[];first=-1;chosen=-1;aa=0.;gain=0.
    qtime=verifytime=0.
    def score(c):
        nonlocal jvps,qtime
        if c['id'] not in cache:
            ts=time.perf_counter();q=qcalc(ctx,c);qtime+=time.perf_counter()-ts;jvps+=1
            al=alpha(c,q,ctx['h']);cache[c['id']]={'alpha':al,'score':c['p']*al}
        return cache[c['id']]
    def check(c,al):
        nonlocal checks,verifytime
        ts=time.perf_counter();r=actual(ctx,c,al);verifytime+=time.perf_counter()-ts;checks+=1;return r
    remaining=[c for c in ctx['cs'] if c['p']>0]
    if method in ['tide_full','oracle']:
        for c in remaining:score(c)
    if method=='oracle':
        best=None
        for c in remaining:
            s=score(c)
            if s['alpha']<=0:continue
            r=check(c,s['alpha'])
            if r['pass'] and (best is None or r['gain']>best[0] or (r['gain']==best[0] and c['id']<best[1])):best=(r['gain'],c['id'],s['alpha'])
        if best:gain,chosen,aa=best;first=chosen
    else:
        while remaining:
            if method=='full_gain':
                c=min(remaining,key=lambda c:(-c['p'],c['id']));s=score(c)
            elif method=='tide_full':
                c=min(remaining,key=lambda c:(-score(c)['score'],c['id']));s=score(c)
            else:
                # Only this method's own cache may be consulted. Near ties expand.
                best=None
                for cand in sorted(remaining,key=lambda c:(-c['upper'],c['id'])):
                    if best is not None:
                        tol=1e-14*max(1.,abs(best[0]))
                        if cand['upper']<best[0]-tol:break
                    ss=score(cand)
                    if best is None or (-ss['score'],cand['id'])<(-best[0],best[1]):best=(ss['score'],cand['id'])
                c=next(c for c in remaining if c['id']==best[1]);s=score(c)
            if first<0:first=c['id']
            if s['alpha']>0 and s['score']>0:
                r=check(c,s['alpha'])
                if r['pass']:chosen=c['id'];aa=s['alpha'];gain=r['gain'];break
                reject.append(c['id'])
            remaining=[x for x in remaining if x['id']!=c['id']]
    seconds=time.perf_counter()-start
    norm=0. if chosen<0 else float((aa*ctx['cs'][chosen]['d']).norm())
    return {'method':method,'first_id':first,'final_id':chosen,'alpha':aa,'gain':gain,'normalized_gain':gain/ctx['pv'] if ctx['pv'] else 0.,'displacement_norm':norm,'rejected':reject,'jvps':jvps,'checks':checks,'selection_seconds':seconds,'jvp_seconds':qtime,'verification_seconds':verifytime}

def natural(problem,state):
    t=state['theta'].clone();m=torch.zeros_like(t);v=torch.zeros_like(t);out={}
    for count in range(1,121):
        pv,bs,gp,gs=B.grads(problem,t);g=gp+100*sum(gs);m=.9*m+.1*g;v=.999*v+.001*g*g
        t=t-.001*(m/(1-.9**count))/(torch.sqrt(v/(1-.999**count))+1e-8)
        if count in [10,40,120]:out[count]={'theta':t.clone(),'m':m.clone(),'v':v.clone(),'count':count}
    return out

def selftest():
    assert bound(0,0,0)==1 and bound(-2,1,0)==1 and bound(1,1,0)==0
    assert bound(1,0,0)==0 and abs(bound(2,0,1)-.5)<1e-15
    assert abs(bound(1e10,1,1)-1e-10)<1e-20
    # Direction switch and an equal upper bound, failed winner then second, zero fallback.
    global qcalc,actual
    oq,oa=qcalc,actual
    try:
        def fakeq(ctx,c):return [c['q']]
        def fakeactual(ctx,c,al):return {'pass':c['id'] not in ctx['fail'],'gain':c['p']*al}
        qcalc=fakeq;actual=fakeactual
        cs=[{'id':0,'p':2.,'a':[0.],'q':100.,'upper':2.,'d':torch.ones(1)}, {'id':1,'p':1.,'a':[0.],'q':0.,'upper':1.,'d':torch.ones(1)}]
        ctx={'cs':cs,'h':[1.],'pv':1.,'fail':[]}
        f=select(ctx,'full_gain');a=select(ctx,'tide_full');b=select(ctx,'tide_lazy')
        assert f['first_id']==0 and a['first_id']==b['first_id']==1
        cs[0]['upper']=cs[1]['upper']=2.;a=select(ctx,'tide_full');b=select(ctx,'tide_lazy');assert a['final_id']==b['final_id']==1
        ctx['fail']=[1];a=select(ctx,'tide_full');b=select(ctx,'tide_lazy');assert a['final_id']==b['final_id']==0 and b['rejected']==[1]
        ctx['fail']=[0,1];assert select(ctx,'tide_lazy')['final_id']==-1
    finally:qcalc=oq;actual=oa
    return {'stable_roots':'pass','direction_switch':'pass','equal_upper':'pass','reject_then_second':'pass','zero_fallback':'pass'}

def main():
    OUT.mkdir(exist_ok=True);(OUT/'snapshots').mkdir(exist_ok=True)
    dump(OUT/'selftest.json',selftest());config={'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'bench_sha256':hashlib.sha256((BASE/'bench.py').read_bytes()).hexdigest(),'torch':torch.__version__,'python':sys.version,'platform':platform.platform(),'threads':torch.get_num_threads(),'dtype':str(torch.get_default_dtype()),'lr':.001,'weights':[0,1,100],'multipliers':[1,4],'timing_repeats':3,'feasibility_comparison':'exact <= cap; no shrink; h<0 => zero','score_tie':'exact S then lower direction ID; lazy pruning tolerance 1e-14*max(1,abs(S))'}
    jobs=[];snaprows=[]
    for name in B.CASES:
        for seed in [0,1,2]:
            prob=B.Problem(name,seed);ini=torch.load(BASE/'main'/f'{name}_s{seed}_initial.pt',weights_only=False);end=torch.load(BASE/'main'/f'{name}_s{seed}_always.pt',weights_only=False)
            _,b0=prob.values(ini['theta']);caps=[max(1e-4,2*b) for b in b0]
            delta=float((end['theta']-ini['theta']).norm());nat=natural(prob,ini)
            for snap,state in [('initial',ini),('always',end)]+[(f'natural_{k}',v) for k,v in nat.items()]:
                state={k:v for k,v in state.items() if k in ['theta','m','v','count']};torch.save(state,OUT/'snapshots'/f'{name}_s{seed}_{snap}.pt')
                _,bs=prob.values(state['theta']);snaprows.append({'case':name,'seed':seed,'snapshot':snap,'count':state.get('count',0),'initial_endpoint_distance':delta,'same_initial_endpoint':delta==0,'constraints':bs,'original_caps':caps,'original_eligible':all(b<=cap for b,cap in zip(bs,caps))})
                for mode in (['original'] if snap in ['initial','always'] else ['original','local']):
                    cap=caps if mode=='original' else [max(1e-4,1.1*b) for b in bs]
                    for mult in [1,4]:jobs.append((name,seed,snap,mode,mult,prob,state,cap))
    write(OUT/'snapshots.csv',snaprows);dump(OUT/'config.json',config)
    # Timing first; no audit q values or trial outcomes exist at this point.
    timing=[]
    print('TIMING_START',time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
    for name,seed,snap,mode,mult,prob,state,caps in jobs:
        if snap not in ['initial','always'] or mult!=1:continue
        for method in METHODS:select(prepare(prob,state,caps,mult),method)
        for rep in range(3):
            order=METHODS[rep:]+METHODS[:rep]
            for method in order:
                ts=time.perf_counter();ctx=prepare(prob,state,caps,mult);common=time.perf_counter()-ts;r=select(ctx,method)
                timing.append({'case':name,'seed':seed,'snapshot':snap,'rep':rep,'order':order.index(method),'common_seconds':common,'total_seconds':common+r['selection_seconds'],**r})
    write(OUT/'timing.csv',timing);print('TIMING_END',time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
    candidates=[];decisions=[];pairs=[]
    for name,seed,snap,mode,mult,prob,state,caps in jobs:
        key={'case':name,'seed':seed,'snapshot':snap,'cap_mode':mode,'multiplier':mult};ctx=prepare(prob,state,caps,mult);eligible=all(h>=0 for h in ctx['h']);key['eligible']=eligible
        results=[select(ctx,method) for method in METHODS]
        af,al=results[1:3]
        assert all(af[k]==al[k] for k in ['first_id','final_id','alpha','gain','rejected']), (key,af,al)
        audit={};test=B.Problem(name,991,n=384,nb=48)
        for c in ctx['cs']:
            q=qcalc(ctx,c);aa=alpha(c,q,ctx['h']);r=actual(ctx,c,aa);audit[c['id']]=r
            z=ctx['theta']+aa*c['d'];tp,tb=test.values(z)
            cos=[]
            for other in ctx['cs']:
                cos.append(float(c['d']@other['d']/(c['d'].norm()*other['d'].norm()+1e-300)))
            candidates.append({**key,'direction_id':c['id'],'weight':c['weight'],'pde_before':ctx['pv'],'constraints_before':ctx['bs'],'caps':caps,'h':ctx['h'],'p':c['p'],'a':c['a'],'q':q,'linear_alpha':c['linear_alpha'],'upper':c['upper'],'alpha':aa,'predicted_gain':c['p']*aa,'p_eligible':c['p']>0,'true_pde':r['pde'],'true_constraints':r['constraints'],'true_gain':r['gain'],'true_normalized_gain':r['gain']/ctx['pv'],'true_safe':r['safe'],'true_pass':r['pass'],'test_pde':tp,'test_constraints':tb,'direction_norm':float(c['d'].norm()),'displacement_norm':float((aa*c['d']).norm()),'direction_cosines':cos})
        oracle=results[3]['gain'];fr=results[0];tr=results[1]
        for r in results:decisions.append({**key,'pde_before':ctx['pv'],'reference_gain':oracle,'regret':oracle-r['gain'],'normalized_regret':(oracle-r['gain'])/ctx['pv'],**r})
        fi,ti=fr['first_id'],tr['first_id'];first_diff=fi!=ti
        first_fd=0. if fi<0 else (audit[fi]['gain'] if audit[fi]['pass'] else 0.)
        first_td=0. if ti<0 else (audit[ti]['gain'] if audit[ti]['pass'] else 0.)
        pairs.append({**key,'first_direction_change':first_diff,'final_direction_change':fr['final_id']!=tr['final_id'],'fallback_only_change':not first_diff and fr['final_id']!=tr['final_id'],'full_gain':fr['gain'],'tide_gain':tr['gain'],'paired_difference':tr['gain']-fr['gain'],'normalized_difference':(tr['gain']-fr['gain'])/ctx['pv'],'same_step_first_gain_difference':first_td-first_fd,'same_step_first_normalized_difference':(first_td-first_fd)/ctx['pv'],'no_valid_candidate':oracle==0.,'full_lazy_equivalent':True})
        print('AUDIT',name,seed,snap,mode,mult,flush=True)
    write(OUT/'candidates.csv',candidates);write(OUT/'decisions.csv',decisions);write(OUT/'paired.csv',pairs)
    # Unit of summary is problem and seed; checkpoints are repeated observations.
    summary=[]
    for layer in ['main_initial','main_always','main_pooled','natural_original','natural_local']:
        for mult in [1,4]:
            def filt(r):
                if r['multiplier']!=mult or not r['eligible']:return False
                if layer.startswith('main_'):return r['snapshot'] in (['initial','always'] if layer=='main_pooled' else [layer[5:]]) and r['cap_mode']=='original'
                return r['snapshot'].startswith('natural') and r['cap_mode']==layer[8:]
            ps=[r for r in pairs if filt(r)];changed=[r for r in ps if r['first_direction_change']]
            unit=[]
            for name in B.CASES:
                for seed in [0,1,2]:
                    rr=[r for r in ps if r['case']==name and r['seed']==seed]
                    if rr:unit.append(float(np.mean([r['normalized_difference'] for r in rr])))
            row={'layer':layer,'multiplier':mult,'observations':len(ps),'problem_seed_units':len(unit),'first_changes':len(changed),'final_changes':sum(r['final_direction_change'] for r in ps),'fallback_only_changes':sum(r['fallback_only_change'] for r in ps),'changed_wins':sum(r['paired_difference']>1e-14 for r in changed),'changed_ties':sum(abs(r['paired_difference'])<=1e-14 for r in changed),'changed_losses':sum(r['paired_difference']< -1e-14 for r in changed),'changed_normalized_mean':float(np.mean([r['normalized_difference'] for r in changed])) if changed else None,'unit_mean_normalized_difference':float(np.mean(unit)) if unit else None,'no_valid':sum(r['no_valid_candidate'] for r in ps)}
            for method in METHODS:
                rr=[r for r in decisions if filt(r) and r['method']==method]
                row[method+'_normalized_regret']=float(np.mean([r['normalized_regret'] for r in rr])) if rr else None
                row[method+'_pass_rate']=sum(r['final_id']>=0 for r in rr)/len(rr) if rr else None
            summary.append(row)
    dump(OUT/'summary.json',summary);write(OUT/'summary.csv',summary)
    dump(OUT/'completion.json',{'completed':time.strftime('%Y-%m-%d %H:%M:%S'),'candidates':len(candidates),'decisions':len(decisions),'paired':len(pairs),'timings':len(timing),'full_lazy_equivalence':'all contexts passed'})
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
