import os,json,csv,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplcache'))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
CASES=['elasticity','kovasznay','nls','gray_scott','helmholtz']
data=[];agg=[]
for case in CASES:
    for seed in [0,1,2]:
        starts=[torch.load(ROOT/tag/f'{case}_s{seed}_initial.pt',weights_only=False)['theta'] for tag in ['main','budget','cache_main','cache_budget']]
        assert all(torch.equal(starts[0],s) for s in starts[1:]),(case,seed)
for tag in ['main','budget','cache_main','cache_budget']:
    rows=json.loads((ROOT/tag/'summary.json').read_text());assert len(rows)==(15 if tag.startswith('cache') else 45),(tag,len(rows))
    cfg=json.loads((ROOT/tag/'config.json').read_text());assert hashlib.sha256((ROOT/tag/'source.py').read_bytes()).hexdigest()==cfg['source_sha256']
    assert all(r['violations']==0 for r in rows)
    data.extend([{'tag':tag,**r} for r in rows])
    for case in CASES:
        for method in (['cache'] if tag.startswith('cache') else ['fo','always','lazy']):
            rs=[r for r in rows if r['case']==case and r['method']==method]
            a={'tag':tag,'case':case,'method':method,'n':len(rs)}
            for metric in ['seconds','steps','accepted','checks','jvps','unsafe_trials','test_pde','relative_l2','jvp_seconds','constraint_seconds','pde_trial_seconds']:
                vs=[r[metric] for r in rs];a[metric]=float(np.mean(vs));a[metric+'_sd']=float(np.std(vs,ddof=1))
            a['test_constraint_violation_runs']=sum(any(b>cap for b,cap in zip(r['test_constraints'],r['caps'])) for r in rs)
            agg.append(a)
(ROOT/'aggregate.json').write_text(json.dumps(agg,indent=2))
with (ROOT/'aggregate.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=agg[0]);w.writeheader();w.writerows(agg)
def get(tag,case,method):
    if method=='cache':tag='cache_'+tag
    return next(r for r in agg if r['tag']==tag and r['case']==case and r['method']==method)
summary=[]
for case in CASES:
    a=get('main',case,'always');l=get('main',case,'lazy')
    c=get('main',case,'cache')
    summary.append({'case':case,'jvp_reduction_percent':100*(1-l['jvps']/a['jvps']),'wall_reduction_percent':100*(1-l['seconds']/a['seconds']),'cache_jvp_reduction_percent':100*(1-c['jvps']/a['jvps']),'cache_wall_reduction_percent':100*(1-c['seconds']/a['seconds']),'always_l2':a['relative_l2'],'lazy_l2':l['relative_l2'],'cache_l2':c['relative_l2']})
(ROOT/'savings.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(1,3,figsize=(15,4.5));x=np.arange(5);colors=['#777777','#D55E00','#0072B2']
for method,color,offset in zip(['fo','always','lazy','cache'],colors+['#009E73'],[-.3,-.1,.1,.3]):
    vals=[get('main',c,method) for c in CASES]
    axes[0].bar(x+offset,[r['seconds'] for r in vals],.19,label=method,color=color)
    axes[1].bar(x+offset,[r['jvps'] for r in vals],.19,color=color)
    vals=[get('budget',c,method) for c in CASES]
    axes[2].bar(x+offset,[r['relative_l2'] for r in vals],.19,color=color)
for ax in axes:ax.set_xticks(x,CASES,rotation=25,ha='right')
axes[0].set_ylabel('Seconds / 120 attempted updates');axes[1].set_ylabel('Constraint JVP calls / 120 attempts');axes[2].set_ylabel('Relative L2 after 3-second optimization budget');axes[0].legend()
fig.suptitle('Five-problem on-demand curvature pilot; means over 3 seeds; submitted training constraints checked')
fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=170);fig.savefig(ROOT/'comparison.pdf');plt.close(fig)
def table(tag):
    lines=['|问题|方法|耗时s|尝试/接受步均值|JVP均值|保护试算均值|测试PDE|相对解L2均值±SD|独立保护点越限运行/3|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in agg:
        if r['tag']!=tag:continue
        lines.append(f"|{r['case']}|{r['method']}|{r['seconds']:.3f}|{r['steps']:.1f}/{r['accepted']:.1f}|{r['jvps']:.1f}|{r['checks']:.1f}|{r['test_pde']:.5g}|{r['relative_l2']:.4f}±{r['relative_l2_sd']:.3g}|{r['test_constraint_violation_runs']}|")
    return '\n'.join(lines)
savings=['|问题|lazy JVP减少|lazy 时间减少|cache JVP减少|cache 时间减少|','|---|---:|---:|---:|---:|']
for r in summary:savings.append(f"|{r['case']}|{r['jvp_reduction_percent']:.1f}%|{r['wall_reduction_percent']:.1f}%|{r['cache_jvp_reduction_percent']:.1f}%|{r['cache_wall_reduction_percent']:.1f}%|")
ref=json.loads((ROOT/'reference_check.json').read_text())
report=f'''---
experiment_id: PINN-B-261111-003
date: 2026-11-11
status: completed
scope: five_problem_budget_pilot
---

# 实验内容

本轮执行五类指定方程的按需约束曲率对照：45条固定120次更新轨迹、45条三秒优化预算轨迹，另补30条曲率缓存轨迹，共120条；另有15条30步先导。每问题/方法n=3独立种子。所有已提交训练保护集检查均未越界；独立保护点可能越界，单独报告，不能称连续边界安全。

**主要发现：失败后升级到曲率仅在部分案例有净节省；跨步缓存大幅减少曲率调用，但可能过度保守而停滞。本轮相对解误差多数仍在0.4–1附近，尚未充分求解这五个问题。120条运行中60条在独立保护点上出现至少一个块超出对应训练上限，必须进一步加强保护采样。** 四个正式/探索组的起始参数已逐位核对一致。

三方案：FO是一阶PDE收益排序加真实约束检查、失败后几何回溯；always给全部3个方向预计算约束残差曲率；lazy只在候选的真实保护检查失败后才计算该方向曲率。三者PDE排序均为一阶，不含PDE JVP。真实检查通过后才试算PDE，下降则提交。最多8次每方向试探，Adam提案λ=0/1/100、lr=1e-3；二次安全根留0.99系数裕量。

lazy是“失败后升级”而非预测到危险前就免费知道曲率。节省量需要同时扣除先行检查与失败试算的成本；FO无曲率，但可能产生大量回溯。

# 五个方程与校验

实际方程、参考解、参数、边界详见实验记录.md。线弹性为五输出位移—应力混合平面应变、制造体力；Kovasznay为Re=40不可压稳态流与压力锚；NLS为sech单孤子、解析Dirichlet端点；Gray–Scott为二维周期反应扩散、短时T=2；Helmholtz为k=5带非零边界制造解。

NLS并非经典2sech初值的周期基准；Gray–Scott仅是短时平滑场，不是长时图案形成难例。这些是五类方程上的第一轮可靠性/成本研究，不是最终论文基准的充分求解。

四种解析解的独立自动微分残差校验最大值约1e-15，空间tanh jets中心差分及约束JVP梯度内积校验全部通过。Gray–Scott独立Fourier+RK4参考，32²/dt.05与64²/dt.025的联合加密相对差 {ref['relative_32_dt05_vs64_dt025']:.4g}。由于短时平滑，此差异很小；它只是离散一致性，不是严格误差上界，也不是长时参考解验证。

网络24×24、float64、CPU单线程；48个PDE点，每边16点，瞬态IC32点；评估384个独立内部点和更密边界，Gray参考64²×5时刻。保护预训练最多1200步后共享checkpoint并重置优化器。保护上限逐块为max(1e-4,2×共享初始MSE)，每组实际值保存在summary；不能称所有问题都达到绝对1e-4精度。困难问题的独立泛化与完整训练仍需加密检验。

# 实验过程与预算核算

先五问题单种子30步管线校验，再按种子0/1/2跑固定步数和三秒组。全部串行，避免实验彼此争用CPU；系统其他进程仍未保证独占。每组源代码快照和SHA256匹配。保护集上限同种子同问题跨方法一致。

记录总优化时间、保护检查/JVP/PDE试算分项时间、调用数、拒绝、最终checkpoint和每步CSV。计时包含梯度与优化，排除共享预训练、最终评估和落盘；JVP包含BC/IC的值及导数周期约束。三秒组最多一轮超时，代码有5000次尝试上限，实际次数见表。

固定采样且拒绝后参数/状态不变，会重复相同失败候选。本轮没有预算恢复或切空间候选，因此仍可能停滞；这些重复开销也反映了当前原型的不足。后续应先缓存/识别确定性无进展再比较效率，不能把消除这类浪费全部归因于二阶信息。

# 固定120次更新结果

{table('main')}

# 按需相对全量曲率的节省

{chr(10).join(savings)}

节省百分比为三个种子的均值之比，仅为探索性描述；实际解误差同时列于前表，不把更低成本自动视为更优求解。lazy可通过不同回溯路径改变轨迹，不能假定两者执行相同更新。

# 同三秒优化预算结果

{table('budget')}

# 后续探索：跨步曲率缓存

观察到lazy反复算曲率后，新增cache：按候选λ缓存最近一次q_i/||d||²，新步用当前||d||²缩放；只用于计算试探步长，真实保护检查仍是最终依据。缓存提案失败才重算该方向JVP并更新缓存。没有按年龄失效或严格误差界，方向变化导致缓存不准时可能过度保守；本轮不将其称为安全统计界。cache是在观察主实验后新增的探索，使用同种子再跑各15条，不是独立确认性样本。

{table('cache_main')}

{table('cache_budget')}

该缓存包含算法状态，且可能改变步长/轨迹，必须同时比较解质量；仅以JVP次数更少不能认定改进。原始摘要记录cache_uses。

需同时看解误差、接受步数和独立约束越限，不只看JVP调用减少。不同方程的原始PDE残差和误差尺度不可直接横比。本轮不做小样本显著性主张，不把采样点/候选当独立n。

![比较图]({(ROOT/'comparison.png').as_posix()})

# 如何理解预算恢复与切空间

边界阈值1e-4、当前误差9.5e-5时，余量为5e-6。切空间方向使线性化边界残差近似不变，但有限更新仍可把误差推到上限。若修复把边界降到2e-5，可恢复8e-5余量，之后再推进PDE。这类似先整理出可用空间，再继续推进主任务。

修复不是无成本：可能增加PDE误差，也需要额外计算。因此这一轮按用户优先级不启用修复，也不构造完整边界Jacobian投影，以免混淆按需曲率的预算效果。只在后续确认曲率调度有效后，再单独检验修复是否值得。

# 下一步发展

1. 本轮已补跨步缓存，下一轮加入按状态/方向变化的失效机制，与当前仅失败重算的缓存比较；提交仍真实复核，不将缓存称为统计安全界。
2. 给所有方法相同的无进展检测/失败缓存，避免固定状态重复试算；比较曲率真正新增收益。
3. 扩大BC/IC点数和候选数，明确真实检查便宜与昂贵时的成本分界。当前保护检查包含周期导数，Gray–Scott与Dirichlet问题成本结构不同。
4. 固定绝对目标精度、扩大预训练与内部点、增加自然完整训练；针对本轮独立保护点越限加密边界采样，不用放宽阈值掩盖。
5. 将NLS扩展到经典周期高阶孤子，Gray–Scott扩展长时图案，线弹性加入牵引边界/近不可压情形，Helmholtz提高波数，Kovasznay增加Re与更大域；每次改变困难度单独记录，保留当前可复现基线。

# 参考来源

- [NVIDIA mixed linear elasticity](https://docs.nvidia.com/physicsnemo/25.11/physicsnemo-sym/user_guide/foundational/linear_elasticity.html)
- [DeepXDE Kovasznay](https://deepxde.readthedocs.io/en/v1.13.1/demos/pinn_forward/Kovasznay.flow.html)：参考其Navier–Stokes设置，解析速度使用sin(2pi y)，并以残差校验纠正文档摘要可能的符号笔误。
- [NLS PINN equations](https://www.nature.com/articles/s41598-023-44541-5)：核对方程；本实验初值与边界改为解析单孤子。
- [NVIDIA Gray–Scott reaction diffusion](https://developer.nvidia.com/gpugems/gpugems2/part-iv-general-purpose-computation-gpus-primer/chapter-31-mapping-computational)

本轮价值在于五类物理约束下的调度成本、保护误差与求解进展如何权衡，不依赖宣称组件绝对原创。
'''
(ROOT/'实验报告.md').write_text(report,encoding='utf-8')
p=ROOT/'实验记录.md';s=p.read_text(encoding='utf-8').replace('status: running','status: completed');s=s.split('\n\n## 完成记录')[0]
s+='\n\n## 完成记录\n\n45条固定步数、45条同墙钟、30条缓存轨迹完成，另15条先导。五类方程/空间jets/参数JVP校验通过，已提交训练保护点无越界；独立检查越限如实列在报告。全部参数与配置存于pilot/main/budget，未覆盖旧实验。\n';p.write_text(s,encoding='utf-8')
(ROOT/'README.md').write_text('''# E003 按需边界/初值曲率：五问题先导

阅读实验报告.md和实验记录.md；main、budget各45条轨迹，cache_main、cache_budget各15条，pilot15条。

复现（新tag避免覆盖）：
```
& F:/RA/.venv/Scripts/python.exe F:/RA/constraint_update_plugin_20261111/e003_on_demand_five/bench.py --tag reproduction --steps 120
```

正式配置快照见main/source.py与budget/source.py；report.py重建本轮报告和图。使用现有CPU PyTorch、NumPy、Matplotlib，无新增安装。Gray参考保存在gray_reference.npz，生成和加密检查记录在reference_check.json。首轮使用相对起点保护阈值，不等价于固定物理精度或完整求解。
''',encoding='utf-8')
manifest=[]
for p in ROOT.rglob('*'):
    if p.is_file() and '.mplcache' not in p.parts and '__pycache__' not in p.parts and p.name not in ['manifest.json','report_output.json']:manifest.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
print(json.dumps({'runs':len(data),'savings':summary,'budget':[r for r in agg if r['tag']=='budget']},indent=2))
