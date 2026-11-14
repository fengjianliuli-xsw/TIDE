"""Offline classify failed terminal candidates, including exact zero float64 displacement."""
import converge as C
rows=[]
for path in sorted(C.OUT.glob('*/*.pt')):
    if path.stem not in ['slope_gn','tide_lazy']:continue
    d=C.torch.load(path,weights_only=False)
    if d['stop']!='feasible_step_stagnation':continue
    prob=C.L.Problem(d['case'],d['seed']);review=C.L.Problem(d['case'],10000+d['seed'],192,32)
    ctx=C.L.context(prob,d['state'],d['caps'])
    c,a,log=C.select(ctx,review,d['method']);assert c is None
    checks=[]
    for v in log['attempts']:
        cc=ctx['cs'][v['id']];delta=v['alpha']*cc['d']
        checks.append({**v,'numerical_zero':C.torch.equal(ctx['theta']+delta,ctx['theta']),'delta_norm':float(delta.norm()),'review_failure':any(b>cap for b,cap in zip(v['review'],d['caps'])),'small_failure':any(b>cap for b,cap in zip(v['small'],d['caps'])),'pde_failure':v['pde']>=ctx['pv']})
    rows.append({'case':d['case'],'seed':d['seed'],'method':d['method'],'positive_descent_candidates':sum(c['p']>0 for c in ctx['cs']),'positive_alpha_candidates':sum(v['alpha']>0 for v in log['q'].values()),'checks':checks,'all_attempts':len(checks),'numerical_zero_trials':sum(v['numerical_zero'] for v in checks),'review_failures':sum(v['review_failure'] for v in checks),'small_failures':sum(v['small_failure'] for v in checks),'pde_failures':sum(v['pde_failure'] for v in checks)})
C.L.dump(C.ROOT/'terminal_diagnostics.json',rows)
print('TERMINAL',len(rows),flush=True)
