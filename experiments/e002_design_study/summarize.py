import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplcache'))
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def loadcsv(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
    data=[]
    for tag in ['main','sgdm','continuous','time_budget','hybrid','hybrid_time']:
        path=ROOT/tag/'summary.json'
        if path.exists():
            rs=json.loads(path.read_text());data +=[{'experiment':tag,**r} for r in rs]
            config=json.loads((ROOT/tag/'config.json').read_text())
            assert hashlib.sha256((ROOT/tag/'source.py').read_bytes()).hexdigest()==config['source_sha256']
            assert all(r['violations']==0 for r in rs)
    groups=[]
    for key in sorted(set((r['experiment'],r['freq'],r['optimizer'],r['method']) for r in data)):
        rs=[r for r in data if (r['experiment'],r['freq'],r['optimizer'],r['method'])==key]
        group=dict(zip(['experiment','freq','optimizer','method'],key));group['n']=len(rs)
        for metric in ['test_pde','relative_l2','seconds','checks','jvps','accepted','repairs','false_safe','bc']:
            vals=[r[metric] for r in rs];group[metric]=float(np.mean(vals));group[metric+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.
        groups.append(group)
    (ROOT/'aggregate.json').write_text(json.dumps(groups,indent=2))
    with (ROOT/'aggregate.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=groups[0]);w.writeheader();w.writerows(groups)
    colors={'fo_tangent':'#999999','fo_all_tangent':'#009E73','gn_tangent':'#0072B2','oracle_tangent':'#D55E00','gn_raw':'#CC79A7','gn_repair_raw':'#E69F00'}
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,freq in enumerate([1,3]):
        for method,color in colors.items():
            ts=[loadcsv(ROOT/'main'/f'f{freq}_s{seed}_adam_{method}.csv') for seed in [0,1,2]]
            for row,metric in enumerate(['pde','bc']):
                a=np.array([[float(r[metric]) for r in tr] for tr in ts]);ax=axes[row,col]
                ax.plot(np.arange(1,a.shape[1]+1),np.median(a,axis=0),label=method,color=color)
                ax.fill_between(np.arange(1,a.shape[1]+1),a.min(0),a.max(0),color=color,alpha=.1)
                ax.set_yscale('log');ax.set_xlabel('Attempted updates');ax.set_ylabel(metric+' MSE')
        axes[0,col].set_title(f'Poisson k={freq}, Adam');axes[1,col].axhline(1e-4,color='black',ls='--')
    axes[0,0].legend(fontsize=8);fig.suptitle('Median and seed range; n=3; equal attempted updates, not equal compute')
    fig.tight_layout();fig.savefig(ROOT/'trajectories.png',dpi=170);fig.savefig(ROOT/'trajectories.pdf');plt.close(fig)
    replay=loadcsv(ROOT/'replay'/'replay_candidates.csv');decisions=loadcsv(ROOT/'replay'/'replay_decisions.csv')
    audit=[]
    for estimator in ['fo','gn','isotropic','shuffled']:
        rs=[r for r in replay if r['estimator']==estimator];ds=[r for r in decisions if r['estimator']==estimator]
        valid=[float(r['regret']) for r in ds if r['regret']!='']
        audit.append({'estimator':estimator,'candidates':len(rs),'bc_mae':float(np.mean([float(r['bc_error']) for r in rs])),'false_safe':sum(int(r['false_safe']) for r in rs),'false_reject':sum(int(r['false_reject']) for r in rs),'decisions':len(ds),'oracle_exists':sum(int(r['oracle_exists']) for r in ds),'feasible_pick':sum(int(r['feasible_pick']) for r in ds),'missed':sum(int(r['missed_opportunity']) for r in ds),'conditional_regret_mean':float(np.mean(valid)) if valid else None})
    (ROOT/'replay_aggregate.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({'runs':len(data),'groups':groups,'replay':audit},indent=2))

if __name__=='__main__':main()
