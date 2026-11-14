"""Use only the two established equations before freezing the common protocol."""
import converge as C
rows=[]
for name in ['nls','gray_scott']:
    for method in ['adam','slope_gn','tide_lazy']:
        d=C.torch.load(C.L.OUT/f'{name}_s0'/f'{method}.pt',weights_only=False)
        v=C.L.Problem(name,70000,192,32);p0=v.values(d['thetas'][0])[0]
        inds=sorted(set([0,len(d['thetas'])-1]+list(range(0,len(d['thetas']),50))))
        hh=[C.evaluate(v,d['thetas'][i],d['caps'],p0,i,d['elapsed'][i]) for i in inds]
        rows.append({'case':name,'method':method,'source':'E005 historical pilot only; sampled, not E006 periodic training','history':hh,'observed_target_count':sum(x['target'] for x in hh)})
C.L.dump(C.ROOT/'pilot_prior_two_cases.json',rows)
C.L.dump(C.ROOT/'frozen_config.json',{**C.CONFIG,'sha256':C.hashlib.sha256(C.Path(C.__file__).read_bytes()).hexdigest(),'pilot':'pilot_prior_two_cases.json','freeze_before_new_cases':True})
print('PILOT CHECKED, PROTOCOL FROZEN',flush=True)
