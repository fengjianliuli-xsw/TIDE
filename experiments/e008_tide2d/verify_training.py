"""Independent validation and solution checks from E008 weights, paired with historical records."""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"e006_convergence"))
from independent_metrics import IndependentMetrics,old


def close(a,b):
    np.testing.assert_allclose(a,b,rtol=3e-9,atol=1e-12)


def verify():
    rows=[]
    for case in old.CASES:
        for seed in range(5):
            folder=ROOT/"results"/f"{case}_s{seed}"
            orig=old.OUT/folder.name
            if (orig/"startup_failure.json").exists():
                continue
            met=IndependentMetrics(case,seed)
            ini=torch.load(orig/"initial.pt",weights_only=False)
            p0=met.losses(met.validation,ini["state"]["theta"])[0]
            for mode in ("linear_rank","gn_rank"):
                r=torch.load(folder/(mode+".pt"),weights_only=False)
                caps=np.asarray(r["caps"])
                close(caps,ini["caps"])
                for k in ("theta","m","v"):assert torch.equal(r["initial"][k],ini["state"][k])
                hist=r["validation"]
                t=r["state"]["theta"]
                p,bc,ic=met.losses(met.validation,t)
                close([p,bc,ic],[hist[-1]["pde"],*hist[-1]["constraints"]])
                error=met.relative_error(t);close(error,r["final_metrics"]["relative_l2"])
                pa,ba,ia=met.losses(met.audit,t)
                close([pa,ba,ia],[r["final_metrics"]["audit_pde"],*r["final_metrics"]["audit_constraints"]])
                for h in hist:
                    close(h["pde"]/p0,h["pde_ratio"])
                    assert h["target"]==bool(h["pde_ratio"]<=.1 and max(np.asarray(h["constraints"])/caps)<=1)
                hit=r["target_confirmation"]
                if hit:
                    stop=hit["confirmation"]
                    j=next(i for i,h in enumerate(hist) if h['step']==stop['step'])
                    assert j>=3 and all(v['target'] for v in hist[j-2:j+1])
                rows.append(dict(case=case,seed=seed,method=mode,stop=r["stop"],accepted=r["state"]["count"],
                                 attempted=r["steps"],hit=bool(hit),hit_time=hit["confirmation"]["elapsed"] if hit else "",
                                 seconds=r["train_seconds"],l2=error,valid_pde_ratio=p/p0,final_target=hist[-1]["target"],
                                 valid_bc_ratio=bc/caps[0],valid_ic_ratio=ic/caps[1],audit_bc_ratio=ba/caps[0],audit_ic_ratio=ia/caps[1]))
        print("verified",case,flush=True)
    assert len(rows)==48
    with (ROOT/"独立复算.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


if __name__=="__main__":verify()
