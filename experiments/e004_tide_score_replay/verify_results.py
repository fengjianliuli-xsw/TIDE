"""Independent reconstruction from exported candidates, plus direct physics spot checks."""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

here = Path(__file__).resolve().parent
out = here / "results"
sys.path.insert(0, str(here.parent / "e003_on_demand_five"))
import bench
import torch

def rows(name):
    with (out / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def key(row):
    return tuple(row[k] for k in ("case","seed","snapshot","cap_mode","multiplier"))

def feasible_step(linear, quadratic, margin):
    if min(margin) < 0:
        return 0.
    lo, hi = 0., 1.
    for _ in range(80):
        mid = (lo + hi) / 2
        if all(a*mid + b*mid**2 <= h for a,b,h in zip(linear, quadratic, margin)):
            lo = mid
        else:
            hi = mid
    return lo

candidates = defaultdict(list)
for row in rows("candidates.csv"):
    candidates[key(row)].append(row)
decisions = {(key(r),r["method"]):r for r in rows("decisions.csv")}
assert len(candidates)==240 and len(decisions)==960
checked = direct = 0
for context, group in candidates.items():
    valid = []
    for r in group:
        al = float(r["alpha"])
        a,q,h = [json.loads(r[k]) for k in ("a","q","h")]
        assert math.isclose(al,feasible_step(a,q,h),abs_tol=2e-12,rel_tol=1e-10)
        if float(r["p"])>0:
            assert float(r["predicted_gain"])<=float(r["upper"])+1e-12
        if r["true_pass"]=="True" and float(r["p"])>0:
            valid.append(r)
        checked+=1
    for method,field in [("full_gain","p"),("tide_full","predicted_gain"),("tide_lazy","predicted_gain"),("oracle","true_gain")]:
        best=min(valid,key=lambda r:(-float(r[field]),int(r["direction_id"]))) if valid else None
        decision=decisions[context,method]
        assert int(decision["final_id"])==(int(best["direction_id"]) if best else -1)
        assert float(decision["gain"])==(float(best["true_gain"]) if best else 0.)
    case,seed,snapshot,mode,mult=context
    if seed!="1" or snapshot not in ("initial","always") or mode!="original" or mult!="1":
        continue
    problem=bench.Problem(case,int(seed))
    state=torch.load(out/"snapshots"/f"{case}_s{seed}_{snapshot}.pt",weights_only=False)
    t=state["theta"]
    pv,_,gp,gs=bench.grads(problem,t)
    for r in group:
        grad=gp+float(r["weight"])*sum(gs)
        count=state.get("count",0)+1
        m=.9*state.get("m",torch.zeros_like(t))+.1*grad
        v=.999*state.get("v",torch.zeros_like(t))+.001*grad.square()
        d=-.001*m/(1-.9**count)/(v.sqrt()/math.sqrt(1-.999**count)+1e-8)
        plus,minus=problem.constraints(t+.001*d),problem.constraints(t-.001*d)
        for x,y,q in zip(plus,minus,json.loads(r["q"])):
            assert math.isclose(float(((x-y)/.002).square().sum()),q,rel_tol=2e-6,abs_tol=1e-12)
        loss,bs=problem.values(t+float(r["alpha"])*d)
        assert math.isclose(loss,float(r["true_pde"]),rel_tol=1e-12,abs_tol=1e-12)
        assert all(math.isclose(u,v,rel_tol=2e-12,abs_tol=1e-14) for u,v in zip(bs,json.loads(r["true_constraints"])))
        direct+=1
audit={"contexts":len(candidates),"records":checked,"method_decisions_verified":len(decisions),"independent_step_bisections":checked,"direct_physics_and_finite_difference_checks":direct}
(out/"independent_verification.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
print(audit)
