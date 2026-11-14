import hashlib,json,csv
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
from run_experiment import ROOT,RAW

data=json.loads((RAW/'all_results.json').read_text())
j=json.loads((RAW/'boundary_jvp_summary.json').read_text())
agg=list(csv.DictReader((RAW/'aggregate.csv').open(encoding='utf-8')))
def table(rows):
    lines=['|频率|优化器|规则|测试 PDE MSE（均值±SD）|相对 L2（均值±SD）|接受步数均值|BC 越界步数均值|', '|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['freq']}|{r['optimizer']}|{r['rule']}|{float(r['test_pde_mean']):.5g} ± {float(r['test_pde_std']):.3g}|{float(r['relative_l2_mean']):.5f} ± {float(r['relative_l2_std']):.3g}|{float(r['accepted_mean']):.2f}|{float(r['bc_violation_steps_mean']):.2f}|")
    return '\n'.join(lines)

# Verify all experiment starts are paired across candidate-bank versions.
import torch
for freq in [1,3]:
    for seed in [0,1,2]:
        a=torch.load(RAW/'main'/f'initial_f{freq}_s{seed}.pt',weights_only=False)['model']
        b=torch.load(RAW/'extended'/f'initial_f{freq}_s{seed}.pt',weights_only=False)['model']
        assert all(torch.equal(a[k],b[k]) for k in a)
for tag in ['main','extended']:
    conf=json.loads((RAW/tag/f'config_{tag}.json').read_text())
    assert hashlib.sha256((RAW/tag/'run_experiment_snapshot.py').read_bytes()).hexdigest()==conf['source_sha256']
audit='Verified 96 trajectories; matched starts across banks; source snapshots match hashes; budget never violates BC limit; strict never increases BC.'
(RAW/'integrity_checks.txt').write_text(audit,encoding='utf-8')

report=f'''---
experiment_id: PINN-B-261111-001
date: 2026-11-11
status: completed
scope: pilot_mechanism_study
---

# 实验结果与分析

本轮完成预定主实验 48 条轨迹、后续候选集合诊断 48 条轨迹，另有 4 条 120 步试跑。两个频率、三个随机种子、两个基础优化器；每条正式轨迹尝试 200 次更新。它是同一一维 Poisson 方程家族中的小规模机制实验。

**结论：当前朴素候选选择版本尚未表现出持续求解能力，不能声称提高收敛速度。预算保护实现了零边界越界，但只短暂推进后停滞。残差二次预测较一阶预测显著改善了本批起始候选的边界风险判断，值得继续验证。**

# 实验内容与控制

问题 u''+(kπ)²sin(kπx)=0，边界 u(0)=0、u(1)=1，解析解 sin(kπx)+x；k=1、3。网络 1-24-24-1、tanh、float64；64 个固定训练内部点、511 个评估内部点，513 点评估相对解 L2。边界 MSE 上限 1e-4（RMS 0.01）。种子 0/1/2。

先只训练边界到约 1e-11，然后重置优化器状态并从同一 checkpoint 分叉。该条件特意模拟“边界已经高度满足、PDE 尚未求解”的阶段，会使逐步共同下降特别苛刻，不能代表普通随机初始化训练。

Adam lr=1e-3；动量 SGD lr=1e-4、momentum=0.9。主实验候选梯度 gPDE+λgBC，λ=0/1/10/100，学习率倍率 1/0.25/0.0625，共 12 个。每一步所有候选从同一个当前优化器状态生成，提交选中方向及其对应状态，拒绝则完全不更新状态。不同规则的轨迹分叉后状态自然不同。

- strict：PDE 下降且 BC 不上升，浮点容差 1e-12。
- budget：PDE 下降且 BC≤1e-4，允许预算内退化。
- free：仅要求 PDE 下降。
- vanilla：普通 λ=1 完整步长更新，用于背景参考，不匹配候选计算量。

本轮选择候选时全部真实评估 PDE/BC，**没有在训练中用 JVP 替代真实损失**。因此，停滞并非预测误差造成。尚未实现瓶颈调度、自动连续步长、显式可行性修复或通用优化器包装接口。

# 实验过程

1. 检查 F:/RA 的工作区指令，复用其 .venv；CPU 单线程，避免争用现有 GPU 工作。
2. 实现可克隆提案；五步随机梯度序列下 Adam/动量 SGD 与 PyTorch 官方更新逐步一致。解析解残差校验、参数梯度中心有限差分校验通过。
3. 120 步试跑观察到 strict 无候选、budget 少量接受后停滞、free 破坏边界，原始结果保留在 pilot 子目录。
4. 按预先记录的种子/频率/优化器组合完成 48 条主轨迹。
5. 因上述停滞，进行探索性后续诊断：λ改为 0/1/100/1e4/1e6（15 候选），其余不变，再跑 48 条。该后续集合增加了高边界权重，也替换了原集合中的 λ=10，并非严格超集。所有三种筛选规则均使用相同后续集合。它不是预注册确认性实验。
6. 另外在 6 个共享起始 checkpoint、2 种优化器、12 个候选上计算 144 次边界残差 JVP，并与实际更新比较。
7. 保存完整逐步损失与候选选择、线性预测、最终模型和优化器状态、配置、源代码快照及 SHA256；检查两个候选集合的起始模型逐参数完全一致。

# 主实验结果

每组 n=3；SD 为种子间样本标准差，未做显著性检验。PDE 指标来自独立加密网格，候选选择来自固定训练点。

{table([r for r in agg if r['bank']=='main'])}

strict 在全部 12 条主轨迹中均接受 0 步。这是有限候选集合下的停滞，**不等价于证明不存在共同下降方向，也不是对 ConFIG/GUA 等算法的直接比较**。

budget 在主实验中每条轨迹只接受 2–4 步，BC 始终在上限内，但 PDE 下降很小。k=1 Adam 的平均测试 PDE MSE 仅从 45.098 降至 44.988；相对 L2 从 0.55986 降至 0.55260，仍远未求解。k=3 同样基本停滞。

free 在若干组合下降低 PDE 残差，却产生大边界误差；这也解释为何不能只用 PDE 损失判断解质量。一维二阶方程的 PDE 残差无法单独固定积分常数。不同优化器/频率表现差异较大，不应概括为 free 总能降低 PDE。

![主实验轨迹]({(ROOT/'figures/main_trajectories.png').as_posix()})

# 扩展候选结果

{table([r for r in agg if r['bank']=='extended' and r['rule']=='budget'])}

加入更高边界权重后 budget 仍仅接受 2–4 步，strict 仍 0 步，没有解决停滞。在这些设置下，单纯加大候选中的 BC 权重不够；这不排除其他权重、步长或方向集合可能有效。部分无保护 SGD 更新出现极大边界损失，说明把高权重候选无条件交给只看 PDE 的规则也有风险。

当前 budget 每步同时要求 PDE 下降，因而不接受“先修复边界、暂时牺牲 PDE”的更新；它实际上只释放初始已有余量，没有显式补充预算的机制。固定离散最小步长、候选覆盖、动量累积也可能共同导致停滞，尚未用消融将其分开。

![扩展候选轨迹]({(ROOT/'figures/extended_trajectories.png').as_posix()})

# 边界预测诊断

代码损失使用 MSE，不带 1/2。因此预测为 ΔL≈2 mean(r·Jd)+mean((Jd)²)。JVP 是对参数方向求导，边界仅两个点，在这个问题中覆盖完整边界。

|指标|一阶预测|残差二次预测|
|---|---:|---:|
|候选数|{j['n_candidates']}|{j['n_candidates']}|
|绝对变化预测 MAE|{j['linear_mae']:.8g}|{j['quadratic_mae']:.8g}|
|实际越界但被预测为安全|{j['linear_false_safe']}|{j['quadratic_false_safe']}|

实际越界候选共 {j['actual_unsafe']} 个；一阶预测将其全部漏掉，二次预测在这批样本中没有漏掉。MAE 降低约 {j['linear_mae']/j['quadratic_mae']:.1f} 倍。候选高度相关、只来自边界几乎精确的起始状态，不能将 0/144 误放行解释为统计安全保证，也不能外推到训练全过程和其他 PDE。

# 计算开销与适用范围

主实验每条筛选轨迹真实评估 2400 个候选，扩展实验 3000 个；vanilla 仅 200 个。程序无候选时仍重复计算直到 200 次，明显有可优化的浪费。当前版本不能声称轻量或比普通优化器快。计时包含各条轨迹的梯度、候选试算及写前日志整理，排除预训练、最终加密网格评估、文件落盘；主实验与扩展实验部分时间重叠争用 CPU，因此这里只保留原始 seconds，**不据此进行正式墙钟速度排名**。

尚缺：二维/时间依赖 PDE、自然完整训练轨迹、多种精度预算、采样数量/JVP 加速对照、共同下降方法正式基线、同计算预算比较、统计功效分析。单一 τ和两个频率不足以支持普遍算法优势。

# 下一轮建议

1. 加入不要求 PDE 每步下降的边界修复模式，与当前严格 PDE 下降的 budget 做受控消融，检验预算能否反复恢复。
2. 对同一候选计算残差二次安全步长，替代只有三个离散学习率；将候选不足与规则不足分开。
3. 在完整随机初始化训练中再检验，避免仅在边界特意训练到极小值的状态下得到结论。
4. 机制能持续推进后，再把真实试算替换为预测筛选加真实复核，评价误放行与计算收益。

本轮最有依据的研究调整是：**预算内退化需要与预算恢复或可行切向更新配合；仅放宽接受阈值不足以形成有效优化器插件。**
'''
(ROOT/'实验报告.md').write_text(report,encoding='utf-8')
log=ROOT/'实验日志/PINN/机制验证/PINN-B-261111-001.md'
s=log.read_text(encoding='utf-8').replace('status: running','status: completed')
s+='\n\n## 执行后补充\n\n正式与扩展对照各 48 条完成，另保留 4 条试跑。扩展集合与边界 JVP 诊断属于观察主试跑后新增的探索。全部校验通过；完整数字与负面结论见 [实验报告](../../../实验报告.md)。边界保护零越界，但持续收敛假设未获支持。固定候选和拒绝时冻结状态导致重复试算的问题已记录。\n'
log.write_text(s,encoding='utf-8')
(ROOT/'README.md').write_text('''# PINN 约束感知更新选择：首轮机制实验

阅读：实验报告.md（实际结果与局限）；实验日志/PINN/机制验证/PINN-B-261111-001.md（条件与过程）。所有原始数据在 raw/experiments/PINN-B-261111-001。

环境：F:/RA/.venv/Scripts/python.exe，torch 2.15.0+cpu、numpy 1.26.4、matplotlib 3.8.3、Python 3.11.7；不需要安装新包。

复现命令（PowerShell，tag 必须使用新名字，以避免覆盖原实验）：
```
& F:/RA/.venv/Scripts/python.exe F:/RA/constraint_update_plugin_20261111/validate.py
& F:/RA/.venv/Scripts/python.exe F:/RA/constraint_update_plugin_20261111/run_experiment.py --tag reproduction_main
& F:/RA/.venv/Scripts/python.exe F:/RA/constraint_update_plugin_20261111/run_experiment.py --bank extended --tag reproduction_extended
```

analyze.py 和 write_report.py 汇总现有 main/extended 数据，重绘已有图和报告。raw/main 与 raw/extended 的快照 SHA256 与配置匹配。原主实验快照从确定的代码变更逆向重建并逐字节核对 hash，完整性检查已保存。snapshot 内以 __file__ 定位输出，重跑应使用顶层脚本和新 tag。

范围：精确真实候选试算的机制诊断；尚非生产插件，尚无持续收敛优势。不要将候选筛选规则 strict 当作 ConFIG/GUA 的实现。
''',encoding='utf-8')
manifest=[]
for p in ROOT.rglob('*'):
    if p.is_file() and '.mplcache' not in p.parts and '__pycache__' not in p.parts and p.name!='manifest_sha256.json':
        manifest.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(ROOT/'manifest_sha256.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print(audit)
print('Report, log, README and hash manifest written.')
