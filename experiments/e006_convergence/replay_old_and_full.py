"""Offline work: historical original TIDE metrics and full-v-lazy selection replay."""
import converge as C
import argparse
def historical():
    for name in C.L.CASES:
        for seed in range(5):
            old=C.L.OUT/f'{name}_s{seed}';out=C.OUT/f'{name}_s{seed}';out.mkdir(exist_ok=True)
            path=out/'original_tide_historical.pt'
            if path.exists() or not (old/'initial.pt').exists():continue
            d=C.torch.load(old/'tide_lazy.pt',weights_only=False);v=C.L.Problem(name,70000+seed,192,32);p0=v.values(d['thetas'][0])[0]
            # Actual original TIDE trajectories are short after real stagnation; retain every committed theta.
            hh=[C.evaluate(v,t,d['caps'],p0,i,d['elapsed'][i]) for i,t in enumerate(d['thetas'])]
            streak=0;hit=None
            for h in hh:
                streak=streak+1 if h['target'] else 0
                if streak>=3 and hit is None:hit=h
            r={'case':name,'seed':seed,'method':'original_tide_historical','source':str(old/'tide_lazy.pt'),'validation':hh,'target_three_updates':hit,'target_confirmation':None,'periodic_confirmation_available':False,'final_target':hh[-1]['target'],'stop':d['stop'],'train_seconds':d['train_seconds'],'steps':len(d['history']),'accepted':d['accepted'],'caps':d['caps'],'state':d['state'],'final_metrics':d['final_metrics'],'initial_validation_pde':p0}
            C.torch.save(r,path);C.L.dump(out/'original_tide_historical_summary.json',{k:v for k,v in r.items() if k not in ['state','validation']})
            print('HISTORICAL',name,seed,len(hh),flush=True)

def full_replay():
    rows=[]
    for path in sorted(C.OUT.glob('*/tide_lazy.pt')):
        d=C.torch.load(path,weights_only=False);p=C.L.Problem(d['case'],d['seed']);review=C.L.Problem(d['case'],10000+d['seed'],192,32)
        # All recorded validation snapshots, plus terminal state.
        snaps=d['snapshots']+[{'step':d['steps'],'state':d['state']}]
        seen=set()
        for snap in snaps:
            if snap['step'] in seen:continue
            seen.add(snap['step']);ctx=C.L.context(p,snap['state'],d['caps'])
            st=C.time.perf_counter();cf,af,lf=C.select(ctx,review,'tide_full');tf=C.time.perf_counter()-st
            st=C.time.perf_counter();cl,al,ll=C.select(ctx,review,'tide_lazy');tl=C.time.perf_counter()-st
            fi=-1 if cf is None else cf['id'];li=-1 if cl is None else cl['id']
            match=fi==li and af==al and lf['attempts']==ll['attempts']
            rows.append({'case':d['case'],'seed':d['seed'],'step':snap['step'],'full_id':fi,'lazy_id':li,'full_alpha':af,'lazy_alpha':al,'match':match,'full_jvps':lf['jvps'],'lazy_jvps':ll['jvps'],'full_seconds':tf,'lazy_seconds':tl,'attempts':len(lf['attempts'])})
            assert match,rows[-1]
        C.L.dump(C.ROOT/'full_lazy_replay.json',rows)
        print('FULL_REPLAY',d['case'],d['seed'],len(seen),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');a=ap.parse_args()
    if a.full:full_replay()
    else:historical()
