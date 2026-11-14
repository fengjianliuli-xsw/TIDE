"""Make the matched research comparisons after independent endpoint reconstruction."""
import csv
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT=Path(__file__).resolve().parent
def numbers(rows,field):
    return [float(x[field]) for x in rows]

def main():
    with (ROOT/"acceptance"/"verified_endpoints.csv").open(encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    methods=("adam","slope_gn","tide_lazy")
    cases=list(dict.fromkeys(x["case"] for x in rows))
    lines=["# 收敛实验的独立比较","",
           "每题预置五个种子。FHN s2 沿用启动失败，五题有 24 个可比较的初态。稳定达标要求验证 PDE 降到原值的 10% 且 BC/IC 在容差内，连续三次定期检查。经验平台、候选耗尽停滞和时间删失是三个不同结局。","",
           "|方法|有效配对|曾稳定达标|终点满足目标|经验平台|可行步/微步等停滞|截尾|稳定达标用时中位数（仅达标者，秒）|",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in methods:
        g=[x for x in rows if x["method"]==m]
        hit=[x for x in g if x["hit"]=="True"]
        c=Counter(x["reason"] for x in g)
        times=median(numbers(hit,"hit_seconds")) if hit else None
        lines.append(f"|{m}|{len(g)}|{len(hit)}|{sum(x['final_target']=='True' for x in g)}|{c['empirical_platform']}|{len(g)-c['empirical_platform']-c['time_censored']}|{c['time_censored']}|{times if times is not None else '—'}|")
    lines+=["","各题表中误差是实际观察终点的相对 L2 均值，包含停滞和截尾。原版沿用 E005 同初态。达标个数按 Adam / 斜率加缩步 / TIDE 加缩步列示。","",
            "|题目|配对数|原版 TIDE|加权 Adam|斜率加缩步|TIDE 加缩步|稳定达标|TIDE 各种子提交数|",
            "|---|---:|---:|---:|---:|---:|---|---|"]
    for c in cases:
        gr=[[x for x in rows if x["case"]==c and x["method"]==m] for m in methods]
        a,s,t=gr
        assert [x['seed'] for x in a]==[x['seed'] for x in s]==[x['seed'] for x in t]
        errors=[mean(numbers(g,"rel_l2")) for g in gr]
        hits=[str(sum(x["hit"]=="True" for x in g)) for g in gr]
        lines.append(f"|{c}|{len(t)}|{mean(numbers(t,'original_error')):.5g}|{errors[0]:.5g}|{errors[1]:.5g}|{errors[2]:.5g}|{' / '.join(hits)}|{', '.join(x['accepted'] for x in t)}|")
    lines+=["","## 同初态下的变化","","|题目|TIDE 总提交步数（原版→加缩步）|缩步轮次通过的提交|相对斜率加缩步，终点误差胜/平/负|Adam / TIDE 最终审计 BC 或 IC 越界|","|---|---|---:|---|---|"]
    for c in cases:
        a,s,t=[[x for x in rows if x["case"]==c and x["method"]==m] for m in methods]
        diffs=[float(x["rel_l2"])-float(y["rel_l2"]) for x,y in zip(t,s)]
        counts=[sum(d< -1e-10 for d in diffs),sum(abs(d)<=1e-10 for d in diffs),sum(d>1e-10 for d in diffs)]
        fail=lambda g:sum(max(float(x["audit_bc_ratio"]),float(x["audit_ic_ratio"]))>1 for x in g)
        lines.append(f"|{c}|{sum(numbers(t,'original_steps')):.0f} → {sum(numbers(t,'accepted')):.0f}|{sum(numbers(t,'shrink_commits')):.0f}|{' / '.join(map(str,counts))}|{fail(a)} / {fail(t)}|")
    lines+=["","所有已启动运行都保留在分母。未达平台的终点精度无法用于宣布各方法的最终收敛精度排名。缩步开销已计入各方法训练时间。历次运行的验证点和训练点固定，时钟采样受机器负载影响。逐运行复算、首次确认时精度及约束误差见 acceptance/verified_endpoints.csv。"]
    with (ROOT/"acceptance"/"terminal_margins.csv").open(encoding="utf-8-sig") as f:
        deep=[x for x in csv.DictReader(f) if x["deeper_trial_pass"]=="True"]
    lines+=["","## 停滞原因的事后核查","",
            f"保持全部方向和约束不变，将各方向的原试步继续除以 2 的 13 到 40 次方。{len(deep)}/24 个 TIDE 终态还能找到一个真实可行步子。这些状态的最紧复核块剩余容差占上限的中位数为 {median(numbers(deep,'review_margin_ratio')):.3g}，最紧小探针块的相应中位数为 {median(numbers(deep,'small_margin_ratio')):.3g}。在 {sum(x['all_push_out']=='True' for x in deep)} 个状态上，所有正预测收益且模型允许正步长的候选，对最紧复核块的一阶变化都为正。更深缩步中每个状态的最佳相对训练 PDE 收益中位数为 {100*median(numbers(deep,'max_relative_gain')):.6f}%。这是冻结终态的单步探测，未加入训练。","",
            "下一轮更有辨别力的实验，是在候选集中加入针对最紧复核块的切向或向内方向，保持容差、候选数量和真实复核成本可比，继续用稳定达标率检验。另可用独立审计检查新增方向是否只改善固定保护点。"]
    (ROOT/"独立比较.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
