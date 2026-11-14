from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
agg=json.loads((ROOT/'aggregate.json').read_text());rep=json.loads((ROOT/'replay_aggregate.json').read_text())
def table(tag,methods=None):
    lines=['|频率|优化器|设计|PDE MSE 均值±SD|相对L2均值±SD|接受步均值|用时均值(s)|复核误放行次数均值|','|---|---|---|---:|---:|---:|---:|---:|']
    for r in agg:
        if r['experiment']!=tag or (methods and r['method'] not in methods):continue
        lines.append(f"|{r['freq']}|{r['optimizer']}|{r['method']}|{r['test_pde']:.5g}±{r['test_pde_sd']:.3g}|{r['relative_l2']:.4f}±{r['relative_l2_sd']:.3g}|{r['accepted']:.1f}|{r['seconds']:.3f}|{r['false_safe']:.1f}|")
    return '\n'.join(lines)
data=[]
for tag in ['main','continuous','sgdm','time_budget','hybrid','hybrid_time']:
    data+=json.loads((ROOT/tag/'summary.json').read_text())
assert len(data)==111,len(data)
assert all(r['violations']==0 for r in data)
reptable=['|模型|边界预测MAE|候选误放行数|误拒数|72次决策中可行选择|有机会但未选到可行更新|', '|---|---:|---:|---:|---:|---:|']
for r in rep:
    reptable.append(f"|{r['estimator']}|{r['bc_mae']:.6g}|{r['false_safe']}|{r['false_reject']}|{r['feasible_pick']}|{r['missed']}|")
text=f'''---
experiment_id: PINN-B-261111-002
date: 2026-11-11
status: completed
interpretation: exploratory_not_confirmatory
---

# 本轮结论

本轮完成111条正式/后续训练轨迹，加8条先导试跑，并对自然加权Adam训练的检查点进行等范数方向回放。与上一轮不同，本轮实际使用了参数方向JVP预测来驱动候选选择。

最有价值的进展是：新增边界残差切空间候选后，预算保护可以持续推进；显式边界修复也可恢复预算。二次信息减少部分场景的无效试算，但收益依赖于阶段和候选集合。允许一阶预测检查更多候选后，它也能持续优化，在当前小网络、两端点边界场景中通常更便宜。尚无证据支持二阶选择在所有条件下优于一阶方案。

# 内容和过程

详见实验记录.md。Poisson解析解sin(kπx)+x，k=1/3；三个种子0/1/2；tanh 1-24-24-1；CPU float64；48训练内部点、511独立加密评估点、513点解误差；边界MSE上限1e-4。主要训练从边界预训练模型开始，优化器状态重置；自然训练只用于回放。

候选λ=0/1/100/1e4，四个倍率1/.25/.0625/.015625。每个方法内部当前候选共享同一个优化器状态；方法分叉后状态各自变化。边界切空间方向投影基础优化器实际更新，并同步一阶矩/动量；这不是只更改标量损失权重。

流程：先8方法80步先导；加入FO全复核和FO严格规则后，开展66条Adam、200步主实验；随后12条连续步长/两步修复、12条SGD、12条三秒预算对照，再补6条混合固定步数和3条混合预算对照，均为观察前序现象后进行的探索。各组保存启动时源代码副本和SHA256，不覆盖先前结果。

JVP校验：PDE和BC的线性系数分别与参数梯度内积一致；BC二次项与完整两行残差Jacobian乘积一致。每组启动均执行断言。GN模型为mean(r²)的线性项加mean((Jd)²)，不是完整Hessian Taylor模型。

FO/GN对候选预测排序，最多真实复核3个预测可接受候选；FO_all复核全部预测可接受候选。复核需PDE真实下降和边界达标才提交；strict要求BC相对当前值不增加（浮点容差1e-12）。因此提交后无越界并不意味着预测无误。oracle是有限候选真实试算选择器，非全局最优算法。

所有111条轨迹的已提交迭代均未越过BC上限。不存在安全概率保证或连续域推广：本题边界只有两个端点，可全量检查；内部PDE和解仅在加密网格评估。

# 设计一：切空间候选与二次预测

{table('main')}

每格n=3，均值±种子样本标准差，未做显著性检验。

低频k=1时，gn_raw的平均测试PDE约44.98，仍停滞；gn_tangent约22.57；oracle_tangent和fo_all_tangent约15.18。说明候选覆盖改善是明确的工程进展，但GN当前的有限复核版本还没有匹配全部真实试算的结果。

高频k=3时，gn_tangent测试PDE约3157.68，优于起始约3944.70；但平均相对解L2约0.778，高于初始约0.691。这表明内部残差下降并不足以说明离散PINN解误差改善，尤其训练点有限、优化尚未充分时。不能只挑选低频结果或PDE损失作为成功证据。

gn_tangent与fo_all_tangent的对比保持同一候选库，但复核预算不同；它回答能否用更多便宜复核替代JVP，而非纯信息因果消融。fo_tangent的停滞部分来自最多3次复核反复落在不安全候选上，因此不能据此称一阶方法本质不可用。严格共同下降在这一近乎精确BC初态下很苛刻；此处strict不是ConFIG/GUA实现。

![曲线]({(ROOT/'trajectories.png').as_posix()})

# 设计二：预算恢复、连续步长与两步修复

主实验gn_repair_raw在无推进候选时，允许先降低BC、暂时增大PDE。200步中能够持续接受更新，但最终PDE改善不大。该算法会重置动量，其收益/代价与修复本身尚未分离。

连续版本gn_cont_tangent用每方向二次模型计算BC允许的正根，并与PDE二次最小点及完整步长取较小值，再测试该值及1/2、1/4、1/16倍。它不是任意精度的一维全局线搜索。

pair_oracle_raw先试一个基础优化器更新，再在新位置计算一次最小范数BC残差修复。仅提交修复后PDE比周期起点下降且BC达标的整个周期；状态重置。内部试探可越界，保证只针对提交端点。它是局部两步事务原型，使用精确候选试算，不依赖GN预测排序；不能将其收益归因于二阶选择。

{table('continuous')}

两步修复、切空间、连续步长分别是值得发展的设计轴。当前结果不支持把它们全部堆叠成一个复杂方法；应先从同成本表现和鲁棒性选择最简组合，再做状态重置、梯度投影和修复频率消融。

# 设计三：跨优化器小组

{table('sgdm')}

这里仅验证一个频率、三个种子，说明接口与设计可用于动量SGD，但不等于普适兼容性或普遍优势。Adam/SGD学习率不同且未做等预算调参，不能据此排名两种基础优化器。

# 设计四：同墙钟预算探索

每次优化环节预算3秒，最多5000次尝试；完整候选评估造成最多一轮的超时。预训练和最终加密评估不计入，初次编译/冷启动未正式控制，系统未保证独占；仅为同机探索性比较。各组按同一程序串行执行。

{table('time_budget')}

应从这组比较判断“额外JVP值得吗”，不能由相同步数结果直接声称更快。两端点BC真实试算非常便宜，这恰好是FO多复核较有利的场景；更复杂边界和更多候选可能改变结论，需以后实测。

# 设计五：自然检查点、等范数方向回放

加权Adam (PDE+100BC)从随机初始化训练300步，保存步0/30/100/300处参数和真实Adam状态；两个频率、三个种子，共24个条件检查点。每点使用5个方向，归一化到半径0.001/0.01/0.05；72个选择问题。对于未达标BC，回放采用不比当前更差的上限；已达标用1e-4。需要按达标状态分层后再外推。

FO、GN、统一曲率、循环打乱曲率使用完全相同的候选实际更新。统一曲率取当前候选GN曲率均值，是带有当前曲率信息的机制控制，不是零额外计算的基线；打乱只做一个固定循环置换，未对多种打乱重复。回放规范化后的方向不提交，不研究其优化器状态一致性。

{chr(10).join(reptable)}

每模型有360个候选评估，72个选择问题，FO与GN在72次回放中选中的方向身份完全相同（包括无选择），均选到54个实际可行更新、漏掉18个机会。GN的边界预测误差可更小，但当前数据没有显示可行选择次数增加。统一/打乱曲率表现较差，提示曲率对应关系不能随意替换，但**不能仅据此宣称GN相对FO具有方向优势**。条件regret仅针对选中真实可行候选计算，原始值保存在replay_aggregate.json，不能忽略失败样本只比较这个均值。

三个随机种子才是主要独立重复层；候选、半径和检查点高度相关，不把360或72当独立n。自然回放只覆盖一种训练配方，当前没有确认性统计推断。

# 设计六：只在保护项使用二阶信息

在观察到全PDE JVP成本后新增混合方案：PDE用已有梯度内积排序，BC用两端点残差JVP计算线性及二次项。其余切空间候选、最多三次复核和精度预算相同。先做两频率三种子200步，再做低频三种子三秒预算。两组均是事后探索，不能当独立确认结果。

{table('hybrid')}

{table('hybrid_time')}

两个频率、三个种子的混合版本最终测试PDE与全GN逐条完全相同，平均用时约1.78/1.85秒，全GN约3.25/3.19秒。该结果支持把二阶信息集中于保护项；但它没有建立方向排序收益。同三秒预算下混合版平均相对L2约0.344，FO_all约0.218，当前仍不占优。单独BC JVP仍需与直接BC前向比较。

# 下一轮具体推进

1. 保留FO_all_tangent作为强、便宜的内部基线；以是否优于它衡量二次模型新增价值。
2. 研究按需二阶信息：先用已有梯度排序，只对预测接近边界预算、多个候选难区分的情况计算BC JVP；避免每步对全部PDE残差做JVP。触发规则需冻结并做消融。
3. 本轮已补BC-only GN，下一轮在新种子和复杂边界上确认其相对全GN与FO_all的质量—成本权衡。
4. 将预算恢复或两步修复与切空间组合，但单独对比状态保留、同步和重置，避免把收益误归因给预测器。
5. 延长低频训练，增加内部随机采样/网格密度；对高频增加独立网格、解误差与局部最大残差诊断，检查离散残差与真实解质量脱节的原因。
6. 扩展二维制造解与Burgers，再讨论插件可扩展性。当前一维结果已明确哪些环节有用，先利用这些证据精化方法。

创新判断基于清晰新增价值：低开销管理可接受退化、状态相关的二阶信息使用和预算恢复。如果最终最佳设计是少量经典组件的有效组合，只要控制充分、收益清楚，同样具有研究价值。

# 复现与原始记录

study.py为当前实现；main/source.py等保存当次实际运行版本，config.json含SHA256。main/、continuous/、sgdm/、time_budget/保存各运行逐步CSV、摘要和checkpoint；replay/保存完整反事实候选与决策。启动tag目录必须不存在，禁止覆盖。源码中少量探索性注释与当前实际行为不一致时，以本报告及保存代码为准，例如repair并未额外截断范数。

本轮尚未完成：正式强外部优化器基线、二维PDE、自然完整训练对照、统计功效规划和大规模JVP效率验证。它是设计筛选研究，不是最终论文证据包。
'''
(ROOT/'实验报告.md').write_text(text,encoding='utf-8')
p=ROOT/'实验记录.md';log=p.read_text(encoding='utf-8').split('\n\n## 完成记录')[0].replace('status: running','status: completed')
log+='\n\n## 完成记录\n\n66条Adam主实验、12条连续步长/两步修复、12条SGD、12条三秒预算实验，以及6条BC二次混合和3条混合预算实验完成，共111条；另8条先导。自然轨迹24条件检查点、72选择问题、每模型360候选。新增连续步长、两步事务、同墙钟组均属探索性后续，代码快照分别保留。全部提交BC检查通过，但完整求解能力尚不足；详细数字与负面结果见实验报告.md。\n'
p.write_text(log,encoding='utf-8')
(ROOT/'README.md').write_text('''# E002: 方向选择与约束预算设计研究

先读实验报告.md，再读实验记录.md。实验是探索性，111条正式/后续轨迹，加8条先导。

环境复用 F:/RA/.venv/Scripts/python.exe，CPU PyTorch，NumPy，Matplotlib；所有数据在本目录。

复现时使用新tag，例如：
```
& F:/RA/.venv/Scripts/python.exe F:/RA/constraint_update_plugin_20261111/e002_design_study/study.py --tag reproduction --steps 200 --methods oracle_tangent,fo_all_tangent,gn_tangent
```

完整原主实验方法列表见main/config.json，不能使用当前脚本默认方法列表来冒充原主实验（后续已新增方法）。每个子目录source.py可核对当次精确源码。summarize.py和report.py重建已有数据的汇总与报告；运行study.py不覆盖已存在tag。
''',encoding='utf-8')
manifest=[]
for path in ROOT.rglob('*'):
    if path.is_file() and not any(s in path.parts for s in ['.mplcache','__pycache__']) and path.name!='manifest.json':
        manifest.append({'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print('Report complete; verified 111 runs and zero committed BC violations.')
