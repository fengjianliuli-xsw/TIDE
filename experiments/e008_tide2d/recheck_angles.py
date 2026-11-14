"""Offline diagnostic: at an online stop, try each distinct runner-up with the same model and gates."""
import csv
import math
import torch
import train as tr


def diagnose(path):
    rec=torch.load(path,weights_only=False)
    state=rec["state"];t=state["theta"];case,seed=rec["case"],rec["seed"];caps=rec["caps"]
    p=tr.L.Problem(case,seed); dense=tr.L.Problem(case,10000+seed,192,32)
    f,g,prop=tr.proposal(p,state);d=prop["d"];full=tr.values(p,dense,t+d,caps,f)
    row=dict(case=case,seed=seed,review_trigger=not full["safe"],eligible_angles=0,
             recovered_rank=0,recovered_halves="",gain_rel=0.,checks=0)
    if full["safe"]:return row
    D,_=tr.basis(p,dense,t,d,caps,full)
    if D is None:return row
    blocks=tr.models(p,dense,t,D);_,ag,gain=tr.ranks(blocks,caps,g,D)
    indices=sorted([k for k in range(64) if float(ag[k]*gain[k])>0],key=lambda k:(-float(ag[k]*gain[k]),k))
    row["eligible_angles"]=len(indices)
    for rank,k in enumerate(indices,1):
        direction=D@tr.C[k]
        for h in range(13):
            z=t+(float(ag[k])*2.**(-h))*direction
            row["checks"]+=1
            v=tr.values(p,dense,z,caps,f)
            if v["pass_step"]:
                assert rank>1, (case,seed,"online winner passes unexpectedly")
                row.update(recovered_rank=rank,recovered_halves=h,gain_rel=(f-v["pde"])/f)
                return row
    return row


if __name__=="__main__":
    rows=[diagnose(p) for p in sorted(tr.OUT.glob("*/gn_rank.pt"))]
    assert len(rows)==24
    with (tr.ROOT/"终态备选角度.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
