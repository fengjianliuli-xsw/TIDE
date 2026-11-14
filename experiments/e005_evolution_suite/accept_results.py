"""Independent E005 acceptance audit from checkpoints and JSONL, including frozen-state rescue."""
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

import longrun as train
from audit_physics import mlp, pde_mse

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "acceptance"
OUT.mkdir(exist_ok=True)


def require_close(x, y, rtol=5e-10, atol=1e-12):
    np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)


def constraints(p, theta):
    def predict(x):
        return mlp(theta, x, p.lo, p.hi, p.nout)
    def derivative(x, axis):
        x = x.clone().requires_grad_()
        y = predict(x)
        return torch.stack([torch.autograd.grad(v.sum(), x, retain_graph=True)[0][:, axis]
                            for v in y.unbind(1)], dim=1)
    def initial(x):
        s = x[:, 0]
        if p.name == "nls":
            amp = 1 / s.cosh()
            return torch.stack((amp * (x[:,1]/2).cos(), amp * (x[:,1]/2).sin()), 1)
        if p.name == "gray_scott":
            bump = (np.pi * (s - 0.5)).cos().square() * (np.pi * (x[:,1] - 0.5)).cos().square()
            return torch.stack((1-.3*bump, .15*bump),1)
        if p.name == "cgle":
            z = .6 + .25 * torch.exp(1j*s) + .1 * torch.exp(-2j*s)
            return torch.view_as_real(z)
        if p.name == "allen_cahn":
            return (s.square() * (np.pi*s).cos())[:,None]
        with h5py.File(train.DATA / "fhn_first.h5") as f:
            group = f["0000"]
            xgrid, ygrid = group["grid/x"][:], group["grid/y"][:]
            interp = RegularGridInterpolator((ygrid,xgrid),group["data"][0].astype(float))
        points = np.clip(x[:,[1,0]].numpy(), [ygrid[0],xgrid[0]], [ygrid[-1],xgrid[-1]])
        return torch.from_numpy(interp(points))
    bc = []
    for axis, (left, right) in enumerate(p.edges):
        if p.name == "nls":
            bc.extend([predict(left)-initial(left), predict(right)-initial(right)])
        elif p.name == "fhn":
            bc.extend([derivative(left,axis), derivative(right,axis)])
        else:
            bc.extend([predict(left)-predict(right), derivative(left,axis)-derivative(right,axis)])
    return [torch.cat(bc).detach().square().mean().item(),
            (predict(p.xic)-initial(p.xic)).detach().square().mean().item()]


def local_rescue(prob, review, r):
    state, caps = r["state"], r["caps"]
    ctx=train.context(prob,state,caps)
    best = None
    for c in ctx["cs"]:
        if c["p"] <= 0:
            continue
        # Start from the score model's own trial step. No change to direction, cap or points.
        a0 = train.R.alpha(c,train.R.qcalc(ctx,c),ctx["h"])
        for halves in range(1,13):
            a=a0*2**(-halves)
            if a <= 0:
                break
            t=state["theta"]+a*c["d"]
            pv, b=prob.values(t)
            rb=[float(z.square().sum()) for z in review.constraints(t)]
            if pv<ctx["pv"] and all(max(u,v)<=cap for u,v,cap in zip(b,rb,caps)):
                gain=(ctx["pv"]-pv)/max(1e-300,ctx["pv"])
                if best is None or gain>best["gain"]:
                    best={"id":c["id"],"halves":halves,"alpha":a,"gain":gain}
                break
    return best


def save(name, rows):
    with (OUT/name).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


def main():
    rows=[]; paired=[]; rescue=[]; checked_steps=0; failures=[]
    for case in train.CASES:
        grid,truth=train.eval_grid(case)
        for seed in range(5):
            folder=train.OUT/f"{case}_s{seed}"
            if (folder/"startup_failure.json").exists() and not (folder/"tide_full.pt").exists():
                failures.append({"case":case, "seed":seed, "detail":json.loads((folder/"startup_failure.json").read_text(encoding="utf-8-sig"))})
                continue
            prob=train.Problem(case,seed)
            test=train.Problem(case,991,384,48)
            review=train.Problem(case,10000+seed,192,32)
            ini=torch.load(folder/"initial.pt",weights_only=False)
            runs={}
            for method in train.METHODS:
                r=torch.load(folder/(method+".pt"),weights_only=False)
                runs[method]=r
                assert all(torch.equal(r["initial"][k],ini["state"][k]) for k in ("theta","m","v"))
                assert r["initial"]["count"]==ini["state"]["count"]==0
                assert r["caps"]==ini["caps"]
                history=[json.loads(l) for l in (folder/(method+".jsonl")).read_text().splitlines()]
                assert history==r["history"]
                budget=runs["adam"]["train_seconds"]
                ix=sum(s["accepted"] and s["elapsed"]<=budget for s in history)
                assert ix==r["budget_index"]
                assert sum(h["accepted"] for h in history)==r["accepted"]==r["state"]["count"]
                assert torch.equal(r["thetas"][0],ini["state"]["theta"])
                assert torch.equal(r["thetas"][-1],r["state"]["theta"])
                assert np.isfinite(r["thetas"].numpy()).all()
                assert np.all(np.diff([h["elapsed"] for h in history])>0)
                assert len(history)==r["accepted"]+int(r["stop"]=="stagnation")
                checked_steps+=len(history)
                if method!="adam":
                    for step in history:
                        for a in step["attempts"]:
                            passes=(a["alpha"]>0 and a["pde"]<step["pde_before"]
                                    and all(max(x,y)<=cap for x,y,cap in zip(a["small"],a["review"],r["caps"])))
                            assert passes==(a["reason"]=="accept")
                        assert step["accepted"]==any(a["reason"]=="accept" for a in step["attempts"])
                if method=="tide_full":
                    state=ini["state"].copy()
                    for step in history:
                        ctx=train.context(prob,state,r["caps"])
                        positive=[c for c in step["candidates"] if c["p"]>0]
                        for c in positive:
                            # Independent bisection including q=0 and nearly active constraints.
                            lo,hi=0.,1.
                            for _ in range(75):
                                mid=(lo+hi)*0.5
                                if all(mid*(a+mid*q)<=h for a,q,h in zip(c["a"],c["q"],ctx["h"])): lo=mid
                                else: hi=mid
                            require_close(c["alpha"],lo,atol=2e-13)
                        ordered=sorted(positive,key=lambda c:(-c["p"]*c["alpha"],c["id"]))
                        assert [c["id"] for c in ordered[:len(step["attempts"])]]==[a["id"] for a in step["attempts"]]
                        if step["accepted"]:
                            c=ctx["cs"][step["final_id"]]
                            state=dict(theta=state["theta"]+step["alpha"]*c["d"],m=c["m"],v=c["v"],count=c["count"])
                            assert torch.equal(state["theta"],r["thetas"][state["count"]])
                    assert torch.equal(state["m"],r["state"]["m"]) and torch.equal(state["v"],r["state"]["v"])
                t=r["thetas"][ix]
                require_close(pde_mse(case,t,test.x,test.lo,test.hi,test.nout),r["budget_metrics"]["audit_pde"])
                b=constraints(test,t)
                require_close(b,r["budget_metrics"]["audit_constraints"])
                pred=torch.cat([mlp(t,x,test.lo,test.hi,test.nout) for x in grid.split(4096)])
                rel=float((pred-truth).norm()/truth.norm())
                require_close(rel,r["budget_metrics"]["relative_l2"])
                positive=grid[:,-1]>0
                poserr=float((pred[positive]-truth[positive]).norm()/truth[positive].norm())
                phase=None
                if case in ("nls","cgle"):
                    z=torch.complex(*pred.unbind(1))
                    ref=torch.complex(*truth.unbind(1))
                    phase=float(((z*ref.conj()).angle().square()*ref.abs().square()).sum().sqrt()/ref.norm())
                rows.append(dict(case=case,seed=seed,method=method,accepted_at_budget=ix,
                                 accepted_total=r["accepted"],terminal_reason=r["stop"],rel_l2=rel,
                                 initial_rel_l2=r["initial_metrics"]["relative_l2"],
                                 initial_bc_ratio=r["initial_metrics"]["audit_constraints"][0]/r["caps"][0],
                                 initial_ic_ratio=r["initial_metrics"]["audit_constraints"][1]/r["caps"][1],
                                 rel_l2_positive_time=poserr,amplitude_weighted_phase_rms_rad=phase,
                                 bc_ratio=b[0]/r["caps"][0],ic_ratio=b[1]/r["caps"][1],
                                 budget_seconds=budget,train_seconds=r["train_seconds"],jvps=r["jvps"]))
            a,b=runs["tide_full"],runs["tide_lazy"]
            assert torch.equal(a["thetas"],b["thetas"]) and a["stop"]==b["stop"]
            for k in ["theta","m","v"]: assert torch.equal(a["state"][k],b["state"][k])
            for x,y in zip(a["history"],b["history"]):
                for k in ["final_id","alpha","attempts"]: assert x[k]==y[k]
            paired.append(dict(case=case,seed=seed,steps=b["accepted"],full_jvp=a["jvps"],lazy_jvp=b["jvps"],
                               full_time=a["train_seconds"],lazy_time=b["train_seconds"]))
            if b["stop"]=="stagnation":
                solution=local_rescue(prob,review,b)
                last=b["history"][-1]
                violations=[[group+"_"+block for group in ("small","review")
                            for block,val,cap in zip(("bc","ic"),attempt[group],b["caps"]) if val>cap]
                            for attempt in last["attempts"]]
                rescue.append(dict(case=case,seed=seed,found=bool(solution),
                                   terminal_denials=json.dumps([s["reason"] for s in last["attempts"]]),
                                   terminal_violated_blocks=json.dumps(violations),
                                   **(solution or dict(id=None,halves=None,alpha=None,gain=0.))))
        print("verified",case,flush=True)
    save("budget_recalculated.csv",rows);save("full_lazy_pairs.csv",paired);save("stall_trials.csv",rescue)
    (OUT/"audit.json").write_text(json.dumps(dict(runs=len(rows),paired_units=len(paired),logged_attempted_steps=checked_steps,
                                                 startup_failures=failures,stalls_checked=len(rescue),recoverable=sum(row["found"] for row in rescue)),indent=2))


if __name__=="__main__":
    main()
