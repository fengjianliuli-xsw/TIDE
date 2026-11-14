"""Single-step paired geometry test on the 24 fixed E006 terminal states."""
from pathlib import Path
import sys
import csv
import math

import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"e006_convergence"))
import converge as e


def model(problem,t,D):
    fn=problem.constraints
    r,j0=torch.func.jvp(fn,(t,),(D[:,0],))
    _,j1=torch.func.jvp(fn,(t,),(D[:,1],))
    blocks=[]
    for y,a,b in zip(r,j0,j1):
        J=torch.stack((a,b),dim=1)
        blocks.append((float(y@y),2*(y@J),J.T@J))
    return blocks


def study(path):
    record=torch.load(path,weights_only=False)
    case,seed=record["case"],record["seed"]
    p=e.L.Problem(case,seed)
    review=e.L.Problem(case,10000+seed,192,32)
    state=record["state"]; t=state["theta"]; caps=record["caps"]
    ctx=e.L.context(p,state,caps)
    anchor=ctx["cs"][0]["d"]
    u=anchor/anchor.norm()
    z=t.detach().requires_grad_()
    losses=[r.square().sum() for r in review.constraints(z)]
    i=min(range(len(caps)),key=lambda i:1-float(losses[i].detach())/caps[i])
    inward=-torch.autograd.grad(losses[i],z)[0]
    v=inward-u*(inward@u); assert v.norm()>1e-12*inward.norm()
    D=torch.stack((anchor,anchor.norm()*v/v.norm()),dim=1)
    F=p.pde(z).square().sum()
    g=torch.autograd.grad(F,z)[0]
    gains=-(g@D)
    blocks=model(p,t,D)+model(review,t,D)
    limits=caps+caps
    rows=[]
    for mode,n,second in [('adam_ray',1,True),('plane_linear',64,False),('linear_rank_gn_step',64,False),('plane_gn',64,True)]:
        best=(0.,0,0.,torch.zeros(2))
        for k in range(n):
            angle=2*math.pi*k/n
            c=torch.tensor([math.cos(angle),math.sin(angle)])
            al=min(e.L.R.bound(float(a@c),max(0.,float(c@G@c)) if second else 0.,max(0.,cap-l))
                   for (l,a,G),cap in zip(blocks,limits))
            predicted=float(gains@c)*al
            if predicted>best[0]:best=(predicted,k,al,c)
        pred,k,al,c=best
        if mode=='linear_rank_gn_step':
            al=min(e.L.R.bound(float(a@c),max(0.,float(c@G@c)),max(0.,cap-l)) for (l,a,G),cap in zip(blocks,limits))
            pred=float(gains@c)*al
        actual=0.;halves=None
        for j in range(13) if pred>0 else []:
            trial=t+(al*2**-j)*(D@c)
            f,cs=p.values(trial)
            rs=[float(r.square().sum()) for r in review.constraints(trial)]
            if f<ctx['pv'] and all(max(a,b)<=lim for a,b,lim in zip(cs,rs,caps)):
                actual=(ctx['pv']-f)/ctx['pv'];halves=j;break
        rows.append(dict(case=case,seed=seed,mode=mode,angle_index=k,alpha=al,pred_rel=pred/ctx['pv'],actual_rel=actual,halves=halves))
    # Independently recompute the JVP at the winning mixed direction, check all four energies.
    c=torch.tensor([math.cos(2*math.pi*k/n),math.sin(2*math.pi*k/n)])
    for prob, offset in ((p,0),(review,2)):
        _,out=torch.func.jvp(prob.constraints,(t,),(D@c,))
        for j,y in enumerate(out):
            torch.testing.assert_close(y@y,c@blocks[offset+j][2]@c,rtol=1e-9,atol=1e-15)
    return rows


if __name__=="__main__":
    rows=[]
    for path in sorted(e.OUT.glob("*/tide_lazy.pt")):
        rows.extend(study(path))
    assert len(rows)==96
    with (ROOT/"frozen_plane.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
