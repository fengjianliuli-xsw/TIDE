import converge as C
import csv,collections,math
ROOT=C.ROOT
LABEL={'adam':'加权Adam','slope_gn':'斜率+GN+缩步','tide_lazy':'按需TIDE+缩步','original_tide_historical':'原版TIDE（E005历史）'}
def main():
    data=[];rows=[]
    for p in C.OUT.glob('*/*.pt'):
        if p.stem not in LABEL:continue
        d=C.torch.load(p,weights_only=False);data.append(d);h=d['validation'][-1];met=d['final_metrics']
        rows.append({'case':d['case'],'seed':d['seed'],'method':d['method'],'stop':d['stop'],'steps':d['steps'],'accepted':d['accepted'],'train_seconds':d['train_seconds'],'stable_target':bool(d.get('target_confirmation')),'target_seconds':(d['target_confirmation']['confirmation']['elapsed'] if d.get('target_confirmation') else ''),'final_target':d['final_target'],'validation_pde_ratio':h['pde_ratio'],'validation_bc_cap_ratio':h['cap_ratios'][0],'validation_ic_cap_ratio':h['cap_ratios'][1],'relative_l2':met['relative_l2'],'audit_bc_cap_ratio':met['audit_constraints'][0]/d['caps'][0],'audit_ic_cap_ratio':met['audit_constraints'][1]/d['caps'][1],'jvps':d.get('jvps',''),'checks':d.get('checks',''),'hard_budget':d.get('hard_budget',120),'unused_budget':d.get('unused_budget','')})
    with (ROOT/'all_runs.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    lines=['# E006：按经验平台、停滞与删失比较的五题实验','','25个计划配对，24个有效配对。FHN s2沿用E005初态启动失败，未放宽cap；其余每条轨迹均从E005完全相同的权重与Adam零动量初态开始。三主方法真实训练，原版TIDE沿用历史；全量TIDE在按需轨迹快照离线回放，不计为额外训练。','','## 协议与边界','','目标为独立验证PDE相对配对起点降至1/10，且验证BC/IC<=原cap，连续三个2秒tick确认。首次确认是里程碑，训练继续至平台、停滞或硬上限。平台使用最近三个10秒非重叠窗口，中位数绝对变化<=固定归一尺度的1%，每窗q90-q10<=2%，至少20有效更新并有可辨位移。120秒为第一阶段，所有该阶段删失者原状态续跑至最多360秒；续跑需满30秒才判平台。详见冻结协议.md。','','所有方法CPU float64、torch单线程，第一阶段三进程按配对分配，续跑阶段六进程按固定删失清单均衡分配（CPU有12物理核/16逻辑核）。两阶段并发度不同，累计用时包含两阶段，各阶段时钟与120秒快照保留，不能把跨阶段wall差直接解释为算法速度比；队列结束时活跃进程减少。训练时钟包括验证与日志，最终参考解测试和回放离线计。真实保护判据为训练PDE严格下降、small及review的BC/IC全部不超过cap；同一缩步倍率下重新竞争全部正步长候选，倍率从1到1/4096。验证点不参与提交方向。','','可行步停滞表示本轮候选与1至1/4096的既定缩步序列耗尽，不证明参数空间所有方向都不可行。“平台”指该指标与观察尺度下经验稳定，不保证到达目标、得到全局最优或满足未采样点。解误差与残差各自报告；原版历史轨迹过短，没有三个间隔2秒的验证tick，因此不伪造稳定达标。','','## 实际终止分布','','|方法|有效运行|稳定达标|终点达标|平台|可行步/微步/数值零停滞|时间删失|','','|---|---:|---:|---:|---:|---:|---:|']
    ej=ROOT/'extension_jobs.json'
    if ej.exists():
        pending=[j for j in C.json.loads(ej.read_text(encoding='utf8')) if not any(d['case']==j[0] and d['seed']==j[1] and d['method']==j[2] and d.get('hard_budget')==360 for d in data)]
        if pending:lines.insert(2,f'**进行中：尚有{len(pending)}条已计划续跑未完成，以下相应项暂显示120秒里程碑，非最终删失结论。**')
    for m in LABEL:
        rr=[r for r in rows if r['method']==m];st=collections.Counter(r['stop'] for r in rr)
        lines.append(f"|{LABEL[m]}|{len(rr)}|{sum(r['stable_target'] for r in rr)}|{sum(r['final_target'] for r in rr)}|{st['empirical_platform']}|{sum(v for k,v in st.items() if 'stagnation' in k or k=='numerical_zero')}|{st['time_censored']}|")
    for m in C.METHODS:
        rr=[r for r in rows if r['method']==m]
        if rr:lines+=['',f"{LABEL[m]}：稳定达标率{sum(r['stable_target'] for r in rr)/len(rr):.1%}，终点达标率{sum(r['final_target'] for r in rr)/len(rr):.1%}，停滞率{sum('stagnation' in r['stop'] or r['stop']=='numerical_zero' for r in rr)/len(rr):.1%}，硬上限删失率{sum(r['stop']=='time_censored' for r in rr)/len(rr):.1%}（有效配对为分母）。"]
    lines+=['','稳定达标和终止原因可同时出现，分母包括未达标运行；不对未达标者虚构达标用时。启动率24/25=96%。','','## 每题每方法结果','','验证PDE和解误差为同方程内种子的中位数，不能跨方程合并为精度排名。平台子集可能不同，删失者保留在表内。','','|方程|方法|n|稳定达标/终点达标|平台/停滞/删失|验证PDE/起点中位|相对L2中位|秒中位|','','|---|---|---:|---:|---:|---:|---:|---:|']
    for n in C.L.CASES:
        for m in LABEL:
            rr=[r for r in rows if r['case']==n and r['method']==m]
            if not rr:continue
            count=lambda pred:sum(pred(r) for r in rr)
            med=lambda key:float(C.np.median([r[key] for r in rr]))
            lines.append(f"|{n}|{LABEL[m]}|{len(rr)}|{count(lambda r:r['stable_target'])}/{count(lambda r:r['final_target'])}|{count(lambda r:r['stop']=='empirical_platform')}/{count(lambda r:'stagnation' in r['stop'] or r['stop']=='numerical_zero')}/{count(lambda r:r['stop']=='time_censored')}|{med('validation_pde_ratio'):.4g}|{med('relative_l2'):.4g}|{med('train_seconds'):.2f}|")
    lines+=['','## 逐种子比较：按需TIDE与同样缩步的斜率法','','|方程|种子|验证PDE比值 lazy/slope|相对L2 lazy/slope|更新数 lazy/slope|终止 lazy/slope|','','|---|---:|---:|---:|---|---|']
    for n in C.L.CASES:
        for seed in range(5):
            pair={r['method']:r for r in rows if r['case']==n and r['seed']==seed}
            if not all(m in pair for m in ['tide_lazy','slope_gn']):continue
            a,b=pair['tide_lazy'],pair['slope_gn']
            lines.append(f"|{n}|{seed}|{a['validation_pde_ratio']/b['validation_pde_ratio']:.4g}|{a['relative_l2']/b['relative_l2']:.4g}|{a['accepted']}/{b['accepted']}|{a['stop']}/{b['stop']}|")
    lines+=['','## 证据与可复现产物','','- all_runs.csv：全部运行与验证、审计、参考解误差；results/逐配对保存最终权重、动量、验证历史、tick快照、逐步尝试及缩步倍率。','- *_confirmed.pt：首次连续三个定期检查达标的权重与动量；未出现者不创建。','- *_120.pt：需要扩展者120秒里程碑的完整状态；最终.pt保留累计轨迹。','- original_tide_historical.pt：E005原版真实轨迹的新验证指标，包含来源路径。','- jvp_crosscheck.json：解析联合参数方向JVP与torch.func.jvp五题数值交叉核对。','- full_lazy_replay.json：逐快照全量/按需选中方向、步长、全部尝试一致性和离线耗时，不将回放耗时视作独立训练。','','## 解释原则','','缩步救回了原版很早失败的部分更新，但是否长期可持续推进由新增轨迹的实际停滞步数与验证趋势决定。方向评分的差异应在每个配对内比较，不能仅根据总体平均精度断言TIDE在所有方程上占优。加权Adam不施加真实cap保护，较低PDE或较低参考解误差需连同最终验证/审计约束一起解释。若硬上限仍有改善，结论为删失，不能当作已收敛结果。']
    lines+=['','## 首次稳定达标确认时间（秒）','','未达标以“—”显示，种子次序为0–4；FHN s2为启动缺失。首次确认后仍继续训练，终点是否保持达标见all_runs.csv。','','|方程|方法|s0|s1|s2|s3|s4|','|---|---|---:|---:|---:|---:|---:|']
    for n in C.L.CASES:
        for m in C.METHODS:
            vals=[]
            for seed in range(5):
                rr=[r for r in rows if r['case']==n and r['seed']==seed and r['method']==m]
                vals.append('缺失' if not rr else (f"{rr[0]['target_seconds']:.2f}" if rr[0]['stable_target'] else '—'))
            lines.append('|'+n+'|'+LABEL[m]+'|'+'|'.join(vals)+'|')
    paired=[]
    for n in C.L.CASES:
        for seed in range(5):
            rr={r['method']:r for r in rows if r['case']==n and r['seed']==seed}
            if all(m in rr for m in C.METHODS):paired.append(rr)
    if paired:
        wins=sum(x['tide_lazy']['validation_pde_ratio']<x['slope_gn']['validation_pde_ratio']-1e-12 for x in paired)
        losses=sum(x['tide_lazy']['validation_pde_ratio']>x['slope_gn']['validation_pde_ratio']+1e-12 for x in paired)
        l2wins=sum(x['tide_lazy']['relative_l2']<x['slope_gn']['relative_l2']-1e-12 for x in paired)
        l2loss=sum(x['tide_lazy']['relative_l2']>x['slope_gn']['relative_l2']+1e-12 for x in paired)
        rescues=sum(x['tide_lazy']['accepted']>x['original_tide_historical']['accepted'] for x in paired if 'original_tide_historical' in x)
        lines+=['','## 本轮观察','',f'按需TIDE相对斜率评分，在{len(paired)}个配对中，终点验证PDE较低{wins}个、较高{losses}个，其余浮点容差内相同；参考解相对L2较低{l2wins}个、较高{l2loss}个。两个终点指标不保证给出相同排名。',f'按需TIDE加缩步相对原版成功更新更多的配对为{rescues}/{len(paired)}；本轮两个保护法最终均耗尽预置候选/缩步搜索，延长了部分轨迹，但没有将它们变成持续下降至验证目标的训练过程。该结论限于本实验的候选族、cap与点集。']
    td=ROOT/'terminal_diagnostics.json'
    if td.exists():
        zz=C.json.loads(td.read_text(encoding='utf8'))
        lines+=['',f"终端{len(zz)}个保护运行共重建{sum(x['all_attempts'] for x in zz)}次失败试探；其中参数float64完全不变{sum(x['numerical_zero_trials'] for x in zz)}次、review越cap {sum(x['review_failures'] for x in zz)}次、small越cap {sum(x['small_failures'] for x in zz)}次、训练PDE不降{sum(x['pde_failures'] for x in zz)}次。失败条件可能重叠，这些数值零试探不是实际提交更新；详见terminal_diagnostics.json。"]
    lines+=['','保护法终端失败主要涉及review点越cap：同一训练集的下降候选即使通过small点，也会被较大复核点集拦下；本实验的候选族与预置缩步序列因此较早耗尽。该机制由逐试探记录支持，不推断为所有可行方向都消失。']
    replay=ROOT/'full_lazy_replay.json'
    if replay.exists():
        re=C.json.loads(replay.read_text(encoding='utf8'));lines+=['',f"全量与按需回放：{len(re)}个保存快照，完全一致{sum(r['match'] for r in re)}个；全量JVP合计{sum(r['full_jvps'] for r in re)}、按需合计{sum(r['lazy_jvps'] for r in re)}。成本为离线顺序回放测量，不能据此反推训练wall加速比。"]
    lines=[line for i,line in enumerate(lines) if not (line=='' and i+1<len(lines) and lines[i+1].startswith('|---'))]
    (ROOT/'中文实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('REPORT',len(rows),flush=True)
if __name__=='__main__':main()
