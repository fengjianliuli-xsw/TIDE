import sys,csv,json,collections,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import train as T
import torch,numpy as np
R=T.ROOT;L=T.L
def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    rows=[];records={};logs={};initial_checks={}
    sources={'linear_rank':T.OUT,'gn_rank':T.OUT,'adam':T.E.OUT,'old_tide':T.E.OUT,'tide_backtrack':T.E.OUT}
    aliases={'old_tide':'original_tide_historical','tide_backtrack':'tide_lazy'}
    for mode,root in sources.items():
        for path in sorted(root.glob(f'*/{aliases.get(mode,mode)}.pt')):
            d=torch.load(path,weights_only=False);records[(d['case'],d['seed'],mode)]=d
            v=d['validation'][-1];m=d['final_metrics'];row=dict(case=d['case'],seed=d['seed'],method=mode,steps=d.get('steps',0),accepted=d.get('accepted',d['state']['count']),seconds=d.get('train_seconds',0),stop=d.get('stop','historical'),stable=d.get('target_confirmation') is not None,final_target=v['target'],validation_pde_ratio=v['pde_ratio'],validation_cap_max=max(v['cap_ratios']),relative_l2=m['relative_l2'],audit_violation=any(m['audit_violations']),triggers=d.get('triggers',0),direct_accepts=d.get('direct_accepts',0),jvps=d.get('jvps',0),checks=d.get('checks',0))
            rows.append(row)
            pair=(d['case'],d['seed'])
            if pair not in initial_checks:
                ini=torch.load(L.OUT/f'{pair[0]}_s{pair[1]}'/'initial.pt',weights_only=False);theta=ini['state']['theta'];caps=ini['caps']
                vb=L.Problem(pair[0],70000+pair[1],192,32).values(theta)[1];ab=L.Problem(pair[0],991,384,48).values(theta)[1]
                initial_checks[pair]=([b>c for b,c in zip(vb,caps)],[b>c for b,c in zip(ab,caps)])
            iv,ia=initial_checks[pair];fv=[b>c for b,c in zip(v['constraints'],d['caps'])];fa=m['audit_violations']
            row.update(initial_validation_violation=any(iv),initial_audit_violation=any(ia),new_validation_violation=any(b and not a for a,b in zip(iv,fv)),new_audit_violation=any(b and not a for a,b in zip(ia,fa)))
            if mode in T.METHODS:
                logs[(d['case'],d['seed'],mode)]=[json.loads(z) for z in path.with_suffix('.jsonl').read_text().splitlines()]
    assert sum(x['method']=='gn_rank' for x in rows)==24
    assert sum(x['method']=='linear_rank' for x in rows)==24
    write_csv(R/'all_runs.csv',rows)
    pairs=[]
    for name in L.CASES:
        for seed in range(5):
            key=(name,seed,'linear_rank')
            if key not in records:continue
            a=records[key];b=records[(name,seed,'gn_rank')]
            pairs.append(dict(case=name,seed=seed,linear_l2=a['final_metrics']['relative_l2'],gn_l2=b['final_metrics']['relative_l2'],gn_minus_linear_l2=b['final_metrics']['relative_l2']-a['final_metrics']['relative_l2'],linear_val=a['validation'][-1]['pde_ratio'],gn_val=b['validation'][-1]['pde_ratio'],linear_steps=a['steps'],gn_steps=b['steps'],linear_seconds=a['train_seconds'],gn_seconds=b['train_seconds']))
    write_csv(R/'paired_runs.csv',pairs)
    offline=json.loads((R/'offline_same_snapshot.json').read_text());same=sum(x['same'] for x in offline)
    txt=['# E008 TIDE-2D 正式训练报告','', '本轮24个有效初态×两方法均真实训练，使用同一PDE Adam提案、向内正交方向、四块GN步长和真实复核，只改变64角的排序依据。五题每题计划5种子，FHN s2沿用启动失败：计划25配对，有效24；失败不静默移出计划分母。','',f'离线选择块对照：E006同一24个冻结终态上，原试步最大相对超限块与当前最小相对余量块相同 {same}/24，不同 {24-same}/24。对照结果未用于修改规则。四块各2列JVP、完整Gram交叉项、独立混合方向JVP、64角向量化与标量根公式、自适应分支零JVP自测全部通过。','', '## 稳定目标与停止','', '|方法|有效运行|曾三连稳定达标|终点达标|经验平台|可行步停滞|时间截尾|', '|---|---:|---:|---:|---:|---:|---:|']
    for mode in sources:
        rr=[x for x in rows if x['method']==mode]
        stable='未测' if mode=='old_tide' else str(sum(x['stable'] for x in rr))
        txt.append(f"|{mode}|{len(rr)}|{stable}|{sum(x['final_target'] for x in rr)}|{sum(x['stop']=='empirical_platform' for x in rr)}|{sum(x['stop'] in ('feasible_step_stagnation','stagnation') for x in rr)}|{sum(x['stop']=='time_censored' for x in rr)}|")
    txt+=['','## 逐题终点误差','', '相对L2为全部有效种子的均值，包含所有停滞/截尾终点。旧TIDE使用E006保存的E005历史轨迹，Adam和加回溯TIDE使用E006实际终点。本轮未重跑历史基线。','', '|题目|配对|旧TIDE|历史Adam|历史TIDE+回溯|linear-rank|GN-rank|GN相对linear胜/平/负|','|---|---:|---:|---:|---:|---:|---:|---|']
    for name in L.CASES:
        pp=[x for x in pairs if x['case']==name];means=[]
        for mode in ['old_tide','adam','tide_backtrack','linear_rank','gn_rank']:
            means.append(f"{np.mean([x['relative_l2'] for x in rows if x['case']==name and x['method']==mode]):.6g}")
        wins=sum(x['gn_minus_linear_l2']< -1e-12 for x in pp);ties=sum(abs(x['gn_minus_linear_l2'])<=1e-12 for x in pp)
        txt.append(f"|{name}|{len(pp)}|"+'|'.join(means)+f'|{wins}/{ties}/{len(pp)-wins-ties}|')
    txt+=['','## 分支与成本','', '|方法|提交步|总迭代|整步直接接受|约束触发|JVP调用|真实复核|累计秒|独立验证终点约束违规|最终更密审计违规|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in T.METHODS:
        rr=[x for x in rows if x['method']==mode]
        txt.append(f"|{mode}|"+'|'.join(str(sum(x[k] for x in rr)) for k in ['accepted','steps','direct_accepts','triggers','jvps','checks'])+f"|{sum(x['seconds'] for x in rr):.3f}|{sum(x['validation_cap_max']>1 for x in rr)}|{sum(x['audit_violation'] for x in rr)}|")
    txt+=['','原版旧TIDE只有历史轨迹和终点复算，没有本轮每2秒三连确认记录，稳定达标记为未测；其原始stagnation名称统一计入停滞。','', '初态在独立点上可能已经超过沿用的small/review cap。下表新增越界按每个BC/IC块判断：初态该块<=cap而终点该块>cap，至少一块如此计一个运行。已有越界持续、消失及同一越界程度变化不混作新增。','', '|方法|初态验证已越界|终点验证越界|新增验证越界|初态审计已越界|终点审计越界|新增审计越界|','|---|---:|---:|---:|---:|---:|---:|']
    for mode in sources:
        rr=[x for x in rows if x['method']==mode]
        counts=[sum(x['initial_validation_violation'] for x in rr),sum(x['validation_cap_max']>1 for x in rr),sum(x['new_validation_violation'] for x in rr),sum(x['initial_audit_violation'] for x in rr),sum(x['audit_violation'] for x in rr),sum(x['new_audit_violation'] for x in rr)]
        txt.append(f'|{mode}|'+ '|'.join(map(str,counts))+'|')
    stoprows=[]
    for (name,seed,mode),ll in logs.items():
        end=ll[-1];stoprows.append(dict(case=name,seed=seed,method=mode,last_trigger=end['trigger'],last_angle=end['angle_index'],last_alpha=end['alpha'],last_predicted_gain=end.get('predicted_gain'),last_attempts=len(end['attempts']),full_pde=end['full_proposal']['pde'],pde_before=end['pde_before'],last_min_model_slack=min(q['loss'] for q in end['gram']) if end['gram'] else None))
    write_csv(R/'terminal_diagnostics.csv',stoprows)
    totalwin=sum(x['gn_minus_linear_l2']< -1e-12 for x in pairs);totaltie=sum(abs(x['gn_minus_linear_l2'])<=1e-12 for x in pairs)
    txt+=['', '## 解释与限制','',f'GN选角相对线性选角的终点参考解误差胜/平/负为 {totalwin}/{totaltie}/{24-totalwin-totaltie}。胜负只是预设终点的配对描述；稳定目标与平台状态应与终点误差共同解释，不能把较小终点误差表述为最终收敛精度优势。', '', '训练时始终以small和review原上限作严格真实复核。独立验证及最终审计点未参与选角，故其越界必须单列；固定保护点安全不能外推到全域。固定64角、单个获选角及12次减半耗尽称本协议可行步停滞，不证明数学上不存在任何可行方向。', '', '120秒是统一里程碑；全部120秒截尾均要求真实续跑至累计360秒或预设停止。本轮若全在120秒前停滞，则不存在应续跑的截尾对象，不能通过无更新等待补足时长。首次目标确认依每2秒三连定期验证；短于三tick的停滞轨迹不能视作已稳定达标。', '', '运行资源固定8个单线程worker，两个方法交错均衡分配。累计训练秒包含提案、JVP、复核、日志、验证，不含最终参考解审计。历史E006首阶段3worker、续跑6worker，历史时间只能报告背景，不宜计算本轮相对历史的纯算法加速。', '', '## 可复核产物','', '- 冻结规则：冻结协议.md、frozen_manifest.json；执行实现：train.py。','- 离线对照：offline_same_snapshot.json；JVP/分支自测：selftest.json。','- results/题目_s种子/方法.pt保存初末权重、每tick权重、优化器缓冲、验证历史、确认信息和停止原因；*_120.pt是第一阶段里程碑。','- 同目录逐步jsonl含真实提案/四块损失、角度、GN/linear预测alpha、完整Gram、真实缩步尝试、JVP/核查计数与时间。','- all_runs.csv、paired_runs.csv、terminal_diagnostics.csv提供逐运行、逐配对及停止分支数据。','']
    (R/'正式报告.md').write_text('\n'.join(txt),encoding='utf8')
    print('report complete',len(rows),'rows',len(pairs),'pairs')
if __name__=='__main__':main()
