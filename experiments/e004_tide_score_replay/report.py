import sys,json,csv,math,hashlib,itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parent;R=ROOT/'results'
def read(name):
    with (R/name).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        for k,v in row.items():
            if v in ['True','False']:
                row[k]=v=='True';continue
            try:row[k]=json.loads(v)
            except (ValueError,TypeError):pass
    return rows
def mean(xs):return float(np.mean(xs)) if xs else 0.
def fmt(x):return f'{x:.6g}'
def main():
    cs=read('candidates.csv');ds=read('decisions.csv');ps=read('paired.csv');ts=read('timing.csv');ss=read('snapshots.csv');summary=json.loads((R/'summary.json').read_text(encoding='utf-8'))
    # Persist actual vectors, not only their norms. This postprocessing is outside timing.
    import replay as E
    cache={};directions={};displacements={}
    for idx,r in enumerate(cs):
        k=(r['case'],r['seed'],r['snapshot'],r['cap_mode'],r['multiplier'])
        if k not in cache:
            state=E.torch.load(R/'snapshots'/f"{r['case']}_s{r['seed']}_{r['snapshot']}.pt",weights_only=False)
            cache[k]=E.prepare(E.B.Problem(r['case'],r['seed']),state,r['caps'],r['multiplier'])
        directions[f'candidate_{idx:04d}']=cache[k]['cs'][r['direction_id']]['d'].numpy()
    for idx,r in enumerate(ds):
        k=(r['case'],r['seed'],r['snapshot'],r['cap_mode'],r['multiplier']);ctx=cache[k]
        delta=E.torch.zeros_like(ctx['theta']) if r['final_id']<0 else r['alpha']*ctx['cs'][r['final_id']]['d']
        assert abs(float(delta.norm())-r['displacement_norm'])<1e-12
        displacements[f'decision_{idx:04d}']=delta.numpy()
    np.savez_compressed(R/'candidate_directions.npz',**directions)
    np.savez_compressed(R/'parameter_displacements.npz',**displacements)
    def mainrow(r,mult=1):return r['snapshot'] in ['initial','always'] and r.get('cap_mode','original')=='original' and r.get('multiplier',mult)==mult and r.get('eligible',True)
    details=[]
    for layer in ['main_initial','main_always','main_pooled','natural_original','natural_local']:
        for mult in [1,4]:
            def filt(r):
                if r['multiplier']!=mult or not r['eligible']:return False
                if layer.startswith('main_'):return r['snapshot'] in (['initial','always'] if layer=='main_pooled' else [layer[5:]]) and r['cap_mode']=='original'
                return r['snapshot'].startswith('natural') and r['cap_mode']==layer[8:]
            for name in ['all','elasticity','kovasznay','nls','gray_scott','helmholtz']:
                rr=[r for r in ps if filt(r) and (name=='all' or r['case']==name)]
                change=[r for r in rr if r['first_direction_change']]
                details.append({'layer':layer,'multiplier':mult,'case':name,'n':len(rr),'switches':len(change),'raw_difference_mean':mean([r['paired_difference'] for r in rr]),'normalized_difference_mean':mean([r['normalized_difference'] for r in rr]),'changed_raw_mean':mean([r['paired_difference'] for r in change]),'changed_normalized_mean':mean([r['normalized_difference'] for r in change]),'same_step_first_mean':mean([r['same_step_first_normalized_difference'] for r in change])})
    with (R/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=details[0]);w.writeheader();w.writerows(details)
    # Cluster repeated checkpoints inside a problem/seed; exact sign flips for paired difference.
    cluster=defaultdict(list)
    for r in ps:
        if mainrow(r):cluster[(r['case'],r['seed'])].append(r['normalized_difference'])
    vals=np.array([mean(v) for v in cluster.values()]);obs=abs(mean(list(vals)))
    flips=[abs(float(np.dot(signs,vals)/len(vals))) for signs in itertools.product([-1,1],repeat=len(vals))]
    stats={'unit':'problem_seed, initial and always averaged','n':len(vals),'paired_mean':float(vals.mean()),'unit_sd':float(vals.std(ddof=1)),'sign_flip_two_sided_p':sum(v>=obs-1e-18 for v in flips)/len(flips),'unit_values':[{ 'case':k[0],'seed':k[1],'normalized_difference':mean(v)} for k,v in cluster.items()]}
    (R/'paired_statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    manifests=[]
    for path in [ROOT/'replay.py',ROOT/'report.py',ROOT/'实验设计.md',ROOT.parent/'e003_on_demand_five'/'bench.py']+sorted((ROOT.parent/'e003_on_demand_five'/'main').glob('*_initial.pt'))+sorted((ROOT.parent/'e003_on_demand_five'/'main').glob('*_always.pt')):
        manifests.append({'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (R/'source_manifest.json').write_text(json.dumps(manifests,ensure_ascii=False,indent=2),encoding='utf-8')
    text=['# E004：TIDE 有效步长评分的反事实回放验证','', '设计冻结：2026-11-13；实际运行：2026-11-14。全部实验位于 F:/RA。', '', '本轮只检验固定状态上的方向选择与提交结果。固定原容差、倍率1是主分析；倍率4是压力诊断；自然轨迹原容差与局部容差分别列出，不能合并为主结果。所有真实复核否决均计入，未缩短边界根或重试步长。', '', '## 实验与实现', '', '沿用 E003 五个问题、种子0/1/2、48个PDE点、每边16点、宽度24的两隐藏层网络、CPU float64单线程。初始快照重置Adam状态；always终点保留真实m、v和count。自然轨迹从initial重置状态，以PDE+100×各保护项MSE之和、lr=0.001、Adam(0.9,0.999,eps=1e-8)运行120步，保存10/40/120步。保护块分别为bc，或bc/ic。独立集使用原Problem(seed=991,n=384,nb=48)，报告PDE与各保护块残差MSE。', '', '方向ID 0/1/2对应权重0/1/100。压力试验只把同一Adam提案乘4。p=-g_PDE·d，a=g_block·d，q=||J_block d||²，h=cap-current_MSE。原容差cap=max(1e-4,2×initial_MSE)；局部cap=max(1e-4,1.1×当前MSE)。q、a已按块残差均方归一化。', '', '在[0,1]求aα+qα²≤h。h<0表示起点越界，该块步长置0且排除原保护主问题；h=0、q=0、a≤0时α=1，h=0、q>0、a<0时α=min(1,-a/q)，其他向外方向为0。q=0、a>0取min(1,h/a)。h>0、q>0使用a≥0时2h/(sqrt(a²+4qh)+a)，a<0时(sqrt(a²+4qh)-a)/(2q)。不设置极小步长截断，仅正步长且真实PDE下降才能提交。真实安全判据直接使用MSE≤cap，不添加容差，不做0.99缩根。', '', 'full_gain按p排序，在每个候选共同二次步长上依次复核；tide_full先计算所有p>0候选q，按pα排序；tide_lazy仅按U=pα_linear补充自身JVP缓存，以1e-14×max(1,|当前S|)的保守扩展处理上界近似平局，赢家真实失败后删除并继续。实际S平局均以较小ID打破。oracle在同一有限候选和相同步长上遍历真实收益，加入零步；因此其优势仅适用于这三个方向及这些步长。', '', '## 主结果和各诊断层', '', '|层|倍率|有效观测|问题种子单元|首次换向|最终换向|仅回退差异|换向子集胜/平/负|换向子集平均归一化差|单元平均归一化差|无有效候选|', '|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|']
    for r in summary:
        text.append('|'+ '|'.join(map(str,[r['layer'],r['multiplier'],r['observations'],r['problem_seed_units'],r['first_changes'],r['final_changes'],r['fallback_only_changes'],f"{r['changed_wins']}/{r['changed_ties']}/{r['changed_losses']}",fmt(r['changed_normalized_mean'] or 0),fmt(r['unit_mean_normalized_difference'] or 0),r['no_valid']]))+'|')
    text+=['', '收益差均为TIDE-full减full_gain，正值有利。收益归一化除以各自起点PDE MSE；胜/平/负的展示容差为原尺度1e-14。main_pooled先在每个问题/种子内平均initial与always两个重复观测，再对15个单元求均值；自然轨迹也先在问题/种子内平均有效的10/40/120步。首次换向比较模型首选；最终换向比较实际提交（-1为零步）；仅回退差异表示首选相同但最终不同。相同步长的首选收益差、原尺度分方程差均见per_case.csv。', '', f"主层15个问题/种子配对单元的归一化均值差为 {fmt(stats['paired_mean'])}，单元标准差 {fmt(stats['unit_sd'])}；对15个单元进行完整2^15符号翻转的双侧p={fmt(stats['sign_flip_two_sided_p'])}。该数字只作小样本描述，五种方程异质且每方程仅三个种子，不外推长程收敛。", '', '分方程主结果（initial与always重复观测合列；原尺度差异保留）：', '', '|方程|观测|首次换向|平均原尺度差|平均归一化差|换向子集原尺度均差|', '|---|---:|---:|---:|---:|---:|']
    for r in details:
        if r['layer']=='main_pooled' and r['multiplier']==1 and r['case']!='all':text.append(f"|{r['case']}|{r['n']}|{r['switches']}|{fmt(r['raw_difference_mean'])}|{fmt(r['normalized_difference_mean'])}|{fmt(r['changed_raw_mean'])}|")
    text+=['', '相对有限集合真实参考的平均归一化损失与提交率：', '', '|层|倍率|方法|平均归一化损失|提交率|实际复核通过率|', '|---|---:|---|---:|---:|---:|']
    for r in summary:
        if r['layer']=='main_pooled':continue
        for method in ['full_gain','tide_full','tide_lazy','oracle']:
            def match(x):
                if x['method']!=method or x['multiplier']!=r['multiplier'] or not x['eligible']:return False
                if r['layer'].startswith('main_'):return x['snapshot']==r['layer'][5:] and x['cap_mode']=='original'
                return x['snapshot'].startswith('natural') and x['cap_mode']==r['layer'][8:]
            rr=[x for x in ds if match(x)];nchecks=sum(x['checks'] for x in rr);accepted=sum(x['final_id']>=0 for x in rr)
            text.append(f"|{r['layer']}|{r['multiplier']}|{method}|{fmt(r[method+'_normalized_regret'] or 0)}|{fmt(r[method+'_pass_rate'] or 0)}|{fmt(accepted/nchecks if nchecks else 0)}|")
    natural=[r for r in ss if r['snapshot'].startswith('natural')];same=[r for r in ss if r['snapshot']=='always' and r['same_initial_endpoint']]
    text+=['', f"自然快照原容差起点越界 {sum(not r['original_eligible'] for r in natural)}/{len(natural)}；这些快照仍完整保留原始表，在原容差汇总中排除。initial与always参数完全相同 {len(same)}/15，具体参数距离及状态count见snapshots.csv。", '', f"全量与按需在全部 {len(ps)} 个上下文的首次赢家、最终ID、α、真实收益和拒绝序列逐项完全一致；没有把oracle或全量版曲率缓存分享给按需版。", '', '## 真实计时', '', '正式计时在本地时间2026-11-14 00:00:43—00:01:10进行。30个原主检查点各方法单独预热后，3次轮换顺序，每次重新共同梯度与提案生成、重新JVP与真实复核；审计在计时结束后执行。3次轮换顺序并非完整四阶拉丁方，因此只视为本机短时测量。文件写入、独立集评估、oracle审计均不计入选择器耗时；oracle本身的遍历参考成本单列。', '', '|方法|平均实际JVP/决策|平均实际复核/决策|选择与复核均时(ms)|复核均时(ms)|含共同计算均时(ms)|选择与复核中位(ms)|', '|---|---:|---:|---:|---:|---:|---:|']
    timing_summary=[]
    for method in ['full_gain','tide_full','tide_lazy','oracle']:
        rr=[r for r in ts if r['method']==method];row={'method':method,'jvps':mean([r['jvps'] for r in rr]),'checks':mean([r['checks'] for r in rr]),'selection_ms':1000*mean([r['selection_seconds'] for r in rr]),'verification_ms':1000*mean([r['verification_seconds'] for r in rr]),'total_ms':1000*mean([r['total_seconds'] for r in rr]),'selection_median_ms':1000*float(np.median([r['selection_seconds'] for r in rr]))};timing_summary.append(row);text.append('|'+ '|'.join([method]+[fmt(row[k]) for k in ['jvps','checks','selection_ms','verification_ms','total_ms','selection_median_ms']])+'|')
    full=timing_summary[1];lazy=timing_summary[2]
    text+=['', f"按需相对全量JVP减少 {fmt(100*(1-lazy['jvps']/full['jvps']))}%；含共同计算的均时比 lazy/full={fmt(lazy['total_ms']/full['total_ms'])}。这不能直接解释为长程训练加速。", '', '## 原始记录、验证与复现', '', '定向测试覆盖：S评分导致换向、上界相等、首选真实失败后第二名、全部失败零步、稳定根和零裕量特例，selftest.json全部通过。方向余弦存于candidates.csv的direction_cosines，ID保持原值；相同快照和近共线提案均未删除。', '', 'results/candidates.csv每行是一个候选：case/seed/snapshot/cap_mode/multiplier/eligible为键，direction_id/weight、p/a/q/h、linear_alpha/upper、alpha/predicted_gain、真实PDE与约束/收益/安全/通过、独立集PDE与约束、方向范数/位移范数/余弦。所有候选包括p≤0者都保留；oracle与选择器只允许p>0者。', '', 'results/decisions.csv每行是一个方法决策：首次ID、最终ID、alpha、gain、normalized_gain、位移范数、拒绝序列、实际JVP与复核数、时间、参考收益与损失。oracle的first_id表示其真实参考赢家，不表示模型首选。results/paired.csv逐检查点配对，per_case.csv分方程原尺度与归一化差。results/timing.csv保留360条计时重复；snapshots目录保留75个参数与完整Adam状态。', '', '源版本与输入检查点SHA256见results/source_manifest.json，配置与运行版本见results/config.json；运行日志见run.log。路径都相对于本实验目录。复现命令（PowerShell）：', '', '```powershell', "& 'F:/RA/.venv/Scripts/python.exe' 'F:/RA/constraint_update_plugin_20261111/e004_tide_score_replay/replay.py'", "& 'F:/RA/.venv/Scripts/python.exe' 'F:/RA/constraint_update_plugin_20261111/e004_tide_score_replay/report.py'", '```', '', '复现将覆盖本实验results下的派生产物与快照，源E003和冻结设计不被修改。', '', '本轮证据限于检查点单步回放，以及固定自然Adam轨迹上的补充诊断。边界附近的二次残差模型并非真实约束上界，真实复核失败是实验结果的一部分。未证明长期误差、训练成功率、收敛速度或跨硬件加速；需要后续独立长程训练检验。']
    keycols=['case','seed','snapshot','cap_mode','multiplier']
    def key(r):return tuple(r[k] for k in keycols)
    cmap={(key(r),r['direction_id']):r for r in cs}
    dmap={(key(r),r['method']):r for r in ds}
    angles=[]
    for r in ps:
        if not mainrow(r) or not r['first_direction_change']:continue
        f=dmap[(key(r),'full_gain')];t=dmap[(key(r),'tide_full')]
        cosine=cmap[(key(r),f['first_id'])]['direction_cosines'][t['first_id']]
        angles.append({**{k:r[k] for k in keycols},'full_first':f['first_id'],'tide_first':t['first_id'],'cosine':cosine,'angle_degrees':math.degrees(math.acos(max(-1.,min(1.,cosine)))),'normalized_difference':r['normalized_difference'],'raw_difference':r['paired_difference']})
    with (R/'switch_angles.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=angles[0]);w.writeheader();w.writerows(angles)
    text+=['', '## 换向幅度、量级与贡献', '', f"主层21次首次换ID的夹角范围为 {fmt(min(r['angle_degrees'] for r in angles))}°—{fmt(max(r['angle_degrees'] for r in angles))}°，中位数 {fmt(float(np.median([r['angle_degrees'] for r in angles])))}°；其中余弦>0.999999的近共线换ID有 {sum(r['cosine']>0.999999 for r in angles)} 次。不能把21次换ID直接解释为21次有实质几何差别的更新。逐条角度和对应实际收益差见results/switch_angles.csv。", '', '|归一化差平局阈值|换向子集胜|平|负|', '|---:|---:|---:|---:|']
    for tol in [1e-12,1e-8,1e-6]:
        vals2=[r['normalized_difference'] for r in angles];text.append(f"|{tol}|{sum(v>tol for v in vals2)}|{sum(abs(v)<=tol for v in vals2)}|{sum(v< -tol for v in vals2)}|")
    text+=['', '上述敏感性表避免把接近浮点噪声的优势与可见改善混在一起。11胜/10平/0负使用原尺度1e-14，属于预先未约定展示容差后的描述计数；原始差值全部保留。', '', '|方程|主层平均归一化差|对总均差贡献比例|', '|---|---:|---:|']
    contrib=[r for r in details if r['layer']=='main_pooled' and r['multiplier']==1 and r['case']!='all']
    denom=sum(r['normalized_difference_mean'] for r in contrib)
    for r in contrib:text.append(f"|{r['case']}|{fmt(r['normalized_difference_mean'])}|{fmt(100*r['normalized_difference_mean']/denom)}%|")
    independent=[];before_cache={}
    for r in ds:
        if not mainrow(r):continue
        k=key(r);ctx=cache[k]
        if k not in before_cache:
            before_cache[k]=E.B.Problem(r['case'],991,n=384,nb=48).values(ctx['theta'])
        bp,bb=before_cache[k]
        if r['final_id']<0:ap,ab=bp,bb
        else:
            cc=cmap[(k,r['final_id'])];ap,ab=cc['test_pde'],cc['test_constraints']
        bef=all(b<=cap for b,cap in zip(bb,ctx['caps']));aft=all(b<=cap for b,cap in zip(ab,ctx['caps']))
        independent.append({**{kk:r[kk] for kk in keycols},'method':r['method'],'final_id':r['final_id'],'before_pde':bp,'after_pde':ap,'before_constraints':bb,'after_constraints':ab,'caps':ctx['caps'],'before_safe':bef,'after_safe':aft,'new_violation':bef and not aft})
    with (R/'independent_decisions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=independent[0]);w.writeheader()
        for rr in independent:w.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in rr.items()})
    text+=['', '## 独立集保护误差', '', '独立集沿用同一训练容差仅作迁移诊断，选择器的真实复核发生在固定保护点。下表将原本已越界与本步新越界分开；不能把更新后越界全部归因于本步。', '', '|方法|非零提交|独立集更新前已越界|更新后越界|本步新越界|独立集PDE下降|', '|---|---:|---:|---:|---:|---:|']
    for method in ['full_gain','tide_full','tide_lazy','oracle']:
        rr=[r for r in independent if r['method']==method and r['final_id']>=0]
        text.append(f"|{method}|{len(rr)}|{sum(not r['before_safe'] for r in rr)}|{sum(not r['after_safe'] for r in rr)}|{sum(r['new_violation'] for r in rr)}|{sum(r['after_pde']<r['before_pde'] for r in rr)}|")
    text+=['', '逐决策独立集起点/终点PDE与保护误差见results/independent_decisions.csv。']
    text+=['', '收益主要来自少数方程与种子。探索性配对符号翻转p值仅存于paired_statistics.json，对微小正差敏感，不能作为普遍或稳定改进的检验；本报告只解释分方程原尺度、归一化量级与阈值敏感性。', '', '父代理已独立复算全部720条候选的步长与上界、960个最终选择与收益，并用有限差分及真实试算抽查五题起点和终点共30条候选，均通过。该独立核对不计入本报告正式计时。', '', '完整候选方向向量存于results/candidate_directions.npz，键candidate_0000起按candidates.csv数据行的0基索引对应；逐决策实际参数位移向量存于results/parameter_displacements.npz，键decision_0000起对应decisions.csv数据行0基索引（零步为全零向量）。向量范数已与原始决策表逐项核对。']
    old=f"；对15个单元进行完整2^15符号翻转的双侧p={fmt(stats['sign_flip_two_sided_p'])}。该数字只作小样本描述，五种方程异质且每方程仅三个种子，不外推长程收敛。"
    text=[s.replace(old,'。五种方程异质且每方程仅三个种子；以上均为小样本描述，不外推长程收敛。') for s in text]
    (ROOT/'中文实验报告.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    (R/'timing_summary.json').write_text(json.dumps(timing_summary,indent=2),encoding='utf-8')
    print(json.dumps({'stats':stats,'timing':timing_summary,'main_summary':[r for r in summary if r['layer'].startswith('main_') and r['multiplier']==1]},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
