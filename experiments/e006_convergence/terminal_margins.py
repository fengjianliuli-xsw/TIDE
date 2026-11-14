"""Inspect the dense-review margin and whether more halving is useful after 12 halvings."""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import converge as run

ROOT=Path(__file__).resolve().parent


def investigate(path):
    d=torch.load(path,weights_only=False)
    assert d["stop"]=="feasible_step_stagnation"
    case,seed=d["case"],d["seed"]
    p=run.L.Problem(case,seed)
    dense=run.L.Problem(case,10000+seed,192,32)
    t=d["state"]["theta"]
    ctx=run.L.context(p,d["state"],d["caps"])
    _,bv,_,grads=run.L.B.grads(dense,t)
    caps=np.array(d["caps"])
    slack=(caps-bv)/caps
    limiting=slack.argmin()
    slopes=[]; gains=[]; halves=[]
    for c in ctx["cs"]:
        if c["p"]<=0:continue
        alpha=run.L.R.alpha(c,run.qcalc(ctx,c),ctx["h"])
        if alpha<=0:continue
        slopes.append(float(grads[limiting]@c["d"]))
        for k in range(13,41):
            a=alpha*2**(-k); trial=t+a*c['d']
            if torch.equal(trial,t):break
            loss,bs=p.values(trial)
            if loss>=ctx["pv"] or np.any(np.array(bs)>caps):continue
            rb=[float(v.square().sum()) for v in dense.constraints(trial)]
            if np.any(np.array(rb)>caps):continue
            gains.append((ctx["pv"]-loss)/ctx["pv"]);halves.append(k)
            break
    return dict(case=case,seed=seed,small_margin_ratio=min(np.array(ctx["h"])/caps),
                review_margin_ratio=min(slack),review_limiting=("bc","ic")[limiting],
                n_eligible=len(slopes),all_push_out=bool(slopes) and all(x>0 for x in slopes),
                deeper_trial_pass=bool(gains),max_relative_gain=max(gains,default=0.),first_halves=min(halves,default=None))


if __name__=="__main__":
    rows=[investigate(path) for path in sorted((ROOT/"results").glob("*/tide_lazy.pt"))]
    assert len(rows)==24
    with (ROOT/"acceptance"/"terminal_margins.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print("passed/24",sum(r["deeper_trial_pass"] for r in rows),"median_relative_gain",
          np.median([r["max_relative_gain"] for r in rows if r["deeper_trial_pass"]]))
