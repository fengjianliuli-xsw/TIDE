"""Independently reconstruct each published E006 endpoint and the common target milestone."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from independent_metrics import IndependentMetrics, old

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "acceptance"
OUT.mkdir(exist_ok=True)
METHODS = ("adam", "slope_gn", "tide_lazy")


def close(a,b):
    np.testing.assert_allclose(a,b,rtol=3e-9,atol=1e-12)


def window_stats(hist, now, excess_only=False):
    """Compute disjoint-window statistics without consulting the trainer's platform flag."""
    medians=[]; spreads=[]
    for end in [now-20,now-10,now]:
        samples=[x for x in hist if end-10 < x["elapsed"] <= end]
        if len(samples)<4:
            return None
        values=np.asarray([[s["pde_ratio"],*s["cap_ratios"]] for s in samples])
        if excess_only:
            values[:,1:]=np.maximum(0,values[:,1:]-1)
        medians.append(np.median(values,axis=0))
        spreads.append(np.diff(np.quantile(values,[.1,.9],axis=0),axis=0)[0])
    return np.array(medians),np.array(spreads)


def audit(case, seed):
    name=f"{case}_s{seed}"
    folder=ROOT/"results"/name
    ini=torch.load(old.OUT/name/"initial.pt",weights_only=False)
    metric=IndependentMetrics(case,seed)
    base=metric.losses(metric.validation,ini["state"]["theta"])[0]
    caps=np.asarray(ini["caps"])
    rows=[]
    previous=torch.load(old.OUT/name/"tide_lazy.pt",weights_only=False)
    for method in METHODS:
        r=torch.load(folder/(method+".pt"),weights_only=False)
        assert all(torch.equal(ini["state"][k],r["initial"][k]) for k in ("theta","m","v"))
        assert r["initial"]["count"]==0
        close(caps,r["caps"])
        t=r["state"]["theta"]
        p,b,i=metric.losses(metric.validation,t)
        hist=r["validation"]
        close([p,b,i],[hist[-1]["pde"],*hist[-1]["constraints"]])
        close(base,r["initial_validation_pde"])
        er=metric.relative_error(t)
        close(er,r["final_metrics"]["relative_l2"])
        pa,ba,ia=metric.losses(metric.audit,t)
        close([pa,ba,ia],[r["final_metrics"]["audit_pde"],*r["final_metrics"]["audit_constraints"]])
        for v in hist:
            close(v["pde_ratio"],v["pde"]/base)
            close(v["cap_ratios"],np.array(v["constraints"])/caps)
            assert v["target"]==bool(v["pde_ratio"]<=0.1 and max(v["cap_ratios"])<=1)
        hit=r["target_confirmation"]; hit_error=None
        next_tick=2.; streak=0; first=None
        for v in hist[1:]:
            if v["elapsed"]+1e-12<next_tick:
                continue
            next_tick=2*(int(v["elapsed"]/2)+1)
            streak=streak+1 if v["target"] else 0
            if streak>=3 and first is None:first=v["step"]
        assert first==(hit["confirmation"]["step"] if hit else None), (name,method,first,hit)
        end=hist[-1]["elapsed"]
        if r["stop"]=="empirical_platform":
            med,span=window_stats(hist,end)
            assert end>=60 and np.abs(np.diff(med,axis=0)).max()<=.01 and span.max()<=.02
            assert r["platform"]["effective_updates"]>=20 and r["platform"]["relative_displacement"]>=1e-8
        # Ground truth plays no part in stopping. Measure error drift across the final validation windows.
        snap={x["step"]:x["state"] for x in r["snapshots"]}
        earlier=min(hist,key=lambda v:abs(v["elapsed"]-(end-30)))
        earlier_error=metric.relative_error(snap[earlier["step"]]["theta"]) if earlier["step"] in snap else None
        continued=folder/(method+"_120.pt")
        milestone_error=None
        if continued.exists():
            original=torch.load(continued,weights_only=False)
            assert original["stop"]=="time_censored"
            assert torch.equal(original["state"]["theta"],snap[original["steps"]]["theta"])
            milestone_error=metric.relative_error(original["state"]["theta"])
        if hit:
            stop=hit["confirmation"]
            h=torch.load(folder/(method+"_confirmed.pt"),weights_only=False)
            vp,*vc=metric.losses(metric.validation,h["theta"])
            close([vp,*vc],[stop["pde"],*stop["constraints"]])
            assert vp/base<=.1 and max(np.array(vc)/caps)<=1
            hit_error=metric.relative_error(h["theta"])
        steps=checks=shrunk=0
        with (folder/(method+".jsonl")).open(encoding="utf-8") as f:
            for line in f:
                s=json.loads(line);steps+=1
                assert steps==s["step"]
                if method!="adam":
                    for a in s["attempts"]:
                        ok= a["pde"]<s["pde_before"] and max(np.maximum(a["small"],a["review"])/caps)<=1 and a["alpha"]>0
                        assert bool(ok)==a["pass"]
                        checks+=1
                    if s["accepted"]:
                        assert s["attempts"][-1]["pass"]
                        shrunk+=s["attempts"][-1]["shrink"]<1
        assert steps==r["steps"] and checks==r["checks"]
        terminal={}
        if method!="adam" and not s["accepted"]:
            terminal={"final_trials":len(s["attempts"]),
                      "failed_training_pde":sum(a["pde"]>=s["pde_before"] for a in s["attempts"]),
                      "failed_small":sum(max(np.array(a["small"])/caps)>1 for a in s["attempts"]),
                      "failed_review":sum(max(np.array(a["review"])/caps)>1 for a in s["attempts"])}
        rows.append(dict(case=case,seed=seed,method=method,reason=r["stop"],accepted=r["accepted"],steps=steps,shrink_commits=shrunk,
                         seconds=r["train_seconds"],hit=bool(hit),hit_seconds=hit["confirmation"]["elapsed"] if hit else "",
                         hit_error=hit_error,final_target=hist[-1]["target"],rel_l2=er,pde_ratio=p/base,bc_ratio=b/caps[0],ic_ratio=i/caps[1],
                         audit_bc_ratio=ba/caps[0],audit_ic_ratio=ia/caps[1],jvps=r["jvps"],checks=checks,
                         original_steps=previous["accepted"],original_error=previous["final_metrics"]["relative_l2"]))
        rows[-1].update(error_at_120=milestone_error,error_earlier=earlier_error,error_window_secs=end-earlier["elapsed"])
        for k in ("final_trials","failed_training_pde","failed_small","failed_review"):
            rows[-1][k]=terminal.get(k,"")
    return rows


if __name__=="__main__":
    all_rows=[]
    for case in old.CASES:
        for seed in range(5):
            if (old.OUT/f"{case}_s{seed}"/"startup_failure.json").exists():
                continue
            all_rows.extend(audit(case,seed))
        print("verified",case,flush=True)
    assert len(all_rows)==72
    with (OUT/"verified_endpoints.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=all_rows[0]);w.writeheader();w.writerows(all_rows)
