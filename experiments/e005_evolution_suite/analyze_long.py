"""Post-training error trajectories and paired summaries; never used for selection."""
import longrun as L
import torch,numpy as np,time,csv,json
def point_audit(prob,t,caps):
    audit=L.Problem(prob.name,991,384,48);rr=audit.constraints(t)
    return [{'fraction_point_residual_squared_above_block_MSE_cap':float((r.square()*r.numel()>cap).double().mean()),'max_residual_squared_over_cap':float(r.square().max()*r.numel()/cap)} for r,cap in zip(rr,caps)]
def main():
    start=time.perf_counter();rows=[];pairs=[];curves=[];ablations=[];failures=[json.loads(p.read_text(encoding='utf8')) for p in L.OUT.glob('*_s*/startup_failure.json')]
    for folder in sorted(L.OUT.glob('*_s*')):
        if not (folder/'initial.pt').exists():continue
        name,ss=folder.name.rsplit('_s',1);seed=int(ss);prob=L.Problem(name,seed);ini=torch.load(folder/'initial.pt',weights_only=False);runs={}
        for method in L.METHODS:
            p=folder/(method+'.pt')
            if not p.exists():continue
            r=torch.load(p,weights_only=False);runs[method]=r;ix=r['budget_index'];t=r['thetas'][ix]
            audit=point_audit(prob,t,r['caps']);initial_audit=point_audit(prob,r['initial']['theta'],r['caps'])
            budget=r['train_seconds'] if r['budget'] is None else r['budget'];status='stagnation' if r['stop']=='stagnation' and r['history'][-1]['elapsed']<=budget else ('budget_cutoff' if ix<r['accepted'] or (r['stop']=='stagnation' and r['history'][-1]['elapsed']>budget) else r['stop'])
            row={'case':name,'seed':seed,'method':method,'accepted':r['accepted'],'budget_accepted':ix,'stop':r['stop'],'budget_status':status,'budget_seconds':budget,'train_seconds':r['train_seconds'],'preparation_seconds':ini['preparation']['seconds'],'caps':r['caps'],'initial_metrics':r['initial_metrics'],'budget_metrics':r['budget_metrics'],'initial_point_audit':initial_audit,'budget_point_audit':audit,'jvps':r['jvps']};rows.append(row)
            indices=sorted(set([0,ix,r['accepted']]+list(range(100,r['accepted']+1,100))))
            for j in indices:curves.append({'case':name,'seed':seed,'method':method,'accepted':j,'elapsed':r['elapsed'][j],**L.metrics(prob,r['thetas'][j],r['caps'])})
        if all(m in runs for m in ['tide_full','tide_lazy']):
            a,b=runs['tide_full'],runs['tide_lazy'];n=min(len(a['thetas']),len(b['thetas']));seq=all((x['final_id'],x['alpha'])==(y['final_id'],y['alpha']) for x,y in zip(a['history'],b['history']))
            pairs.append({'case':name,'seed':seed,'common_accepted':n-1,'max_theta_difference':float((a['thetas'][:n]-b['thetas'][:n]).abs().max()),'same_decision_prefix':seq,'same_stagnation':a['stop']==b['stop'] and a['accepted']==b['accepted'],'full_common_jvps':sum(x['jvps'] for x in a['history'][:min(len(a['history']),len(b['history']))]),'lazy_common_jvps':sum(x['jvps'] for x in b['history'][:min(len(a['history']),len(b['history']))])})
        if (folder/'ablations.json').exists():
            review=L.Problem(name,10000+seed,192,32)
            for item in json.loads((folder/'ablations.json').read_text(encoding='utf8')):
                ss=ini['state'] if item['snapshot']=='initial' else runs['adam']['state'];bs=[float(q.square().sum()) for q in prob.constraints(ss['theta'])];rb=[float(q.square().sum()) for q in review.constraints(ss['theta'])];eligible=all(b<=c for b,c in zip(bs,ini['caps'])) and all(b<=c for b,c in zip(rb,ini['caps']))
                ablations.append({'case':name,'seed':seed,**item,'original_eligible':item['eligible'],'eligible':eligible,'posthoc_exclusion':item['eligible'] and not eligible,'small':bs,'review':rb})
    L.dump(L.OUT/'analysis_rows.json',rows);L.dump(L.OUT/'equivalence.json',pairs);L.dump(L.OUT/'error_trajectories.json',curves)
    L.dump(L.OUT/'summary.json',rows);L.dump(L.OUT/'completion.json',{'time':time.strftime('%Y-%m-%d %H:%M:%S'),'runs':len(rows),'paired_units':len(set((r['case'],r['seed']) for r in rows)),'analysis':'analyze_long.py; combined all completed invocations'})
    files=[L.ROOT/'longrun.py',L.ROOT/'analyze_long.py',L.ROOT.parent/'e003_on_demand_five/bench.py',L.ROOT.parent/'e004_tide_score_replay/replay.py',L.DATA/'AC.mat',L.DATA/'fhn_first.h5',L.DATA/'cgle.npz',L.ROOT.parent/'e003_on_demand_five/gray_reference.npz']
    L.dump(L.OUT/'manifest.json',{'attempted_cases':L.CASES,'attempted_seeds':list(range(5)),'successful_units':len(set((r['case'],r['seed']) for r in rows)),'startup_failed_units':len(failures),'torch':torch.__version__,'numpy':np.__version__,'python':L.sys.version,'threads':torch.get_num_threads(),'sha256':{str(p):L.hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()},'note':'config.json records the last runner invocation; this manifest combines all invocations. Pilot excluded; computational methods frozen after duplicate-gradient correction, later fixes confined to initialization and official FHN IC target caching.'})
    L.dump(L.OUT/'ablation_audit.json',ablations)
    L.dump(L.OUT/'startup_failures.json',failures)
    table=[]
    for r in rows:table.append({k:r[k] for k in ['case','seed','method','accepted','budget_accepted','budget_status','budget_seconds','train_seconds','jvps']}|{'relative_l2':r['budget_metrics']['relative_l2'],'components':json.dumps(r['budget_metrics']['components']),'audit_constraints':json.dumps(r['budget_metrics']['audit_constraints']),'audit_excess':json.dumps(r['budget_metrics']['audit_excess']),'initial_audit_violations':json.dumps(r['initial_metrics']['audit_violations']),'audit_violations':json.dumps(r['budget_metrics']['audit_violations'])})
    with (L.OUT/'main_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    lines=['# E005 五题连续实验报告','',f'实际完成 {len(rows)} 条主轨迹；单位为方程×种子。以下数值由保存权重和原始日志生成。','', '## 规程','', 'CPU float64，单线程，24×24 tanh 网络。训练PDE48点；小保护每边16点、IC32点；固定复核PDE192点（方向选择只用其保护每边32点、IC64点）；独立审计PDE384点、每边48点、IC96点。采样固定。先联合小保护和复核保护拟合1200步或总保护损失<1e-5；若1200步复核仍不满足small×2 cap，则每200步检查并继续至最多12000步，最终cap仍按最终small损失计算，之后各方法共享同初态和零moments。每块cap=max(1e-4,2×初始小保护MSE)，所有启动均同时检查小保护与复核。该相对宽限只复现E004保护口径，绝非物理精度目标。','', 'Adam对PDE+100(IC+BC)做单次反传；其1000次更新实测训练时间给出配对预算。其余方法均单次B.grads生成权重0/1/100三个Adam候选，同一状态共享moments；斜率基线按p排序、full与lazy按p×GN步长排序。真实PDE须严格下降，小保护和更密复核MSE须均<=cap，无安全余量和回溯缩步。全部候选失败即自然停滞。全方法最多1000提交；等时间主表选累计耗时<=预算的最后实际提交，停滞维持终态。训练计时包括候选、JVP、所有真实试算和逐步JSONL落盘；参考生成、初态拟合、热身、结果审计、最终保存分别在训练计时外，初态拟合时间逐单元披露。','', '## 等时间结果','', '|题目|方法|已启动种子误差均值|范围|预算内提交中位数|独立BC/IC越界终点数|','|---|---|---:|---|---:|---|']
    for name in L.CASES:
        for m in L.METHODS:
            rr=[r for r in rows if r['case']==name and r['method']==m]
            if not rr:continue
            ee=[r['budget_metrics']['relative_l2'] for r in rr];v=np.array([r['budget_metrics']['audit_violations'] for r in rr]);lines.append(f'|{name}|{m}|{np.mean(ee):.6g}|{min(ee):.6g}–{max(ee):.6g}|{np.median([r["budget_accepted"] for r in rr]):g}|{v.sum(0).tolist()} / {len(rr)}|')
    lines+=['','## 跨集启动与新增超限','','|题目|方法|独立启动BC/IC超限|终点新增BC/IC超限|终点已恢复BC/IC|','|---|---|---|---|---|']
    for name in L.CASES:
        for m in L.METHODS:
            rr=[r for r in rows if r['case']==name and r['method']==m]
            if not rr:continue
            a=np.array([r['initial_metrics']['audit_violations'] for r in rr]);b=np.array([r['budget_metrics']['audit_violations'] for r in rr]);lines.append(f'|{name}|{m}|{a.sum(0).tolist()}|{(~a&b).sum(0).tolist()}|{(a&~b).sum(0).tolist()}|')
    lines+=['','## 逐种子主结果','','|题目|种子|方法|误差|分量误差|预算内/全提交|终止|秒/预算|','|---|---:|---|---:|---|---|---|---|']
    for r in rows:lines.append(f'|{r["case"]}|{r["seed"]}|{r["method"]}|{r["budget_metrics"]["relative_l2"]:.6g}|{[round(x,6) for x in r["budget_metrics"]["components"]]}|{r["budget_accepted"]}/{r["accepted"]}|{r["budget_status"]}|{r["train_seconds"]:.4g}/{r["budget_seconds"]:.4g}|')
    lines+=['','## 配对结论','']
    for name in L.CASES:
        for baseline in ['adam','slope_gn']:
            aa={r['seed']:r for r in rows if r['case']==name and r['method']=='tide_lazy'};bb={r['seed']:r for r in rows if r['case']==name and r['method']==baseline};seeds=sorted(aa.keys()&bb.keys())
            if not seeds:continue
            ds=[aa[s]['budget_metrics']['relative_l2']-bb[s]['budget_metrics']['relative_l2'] for s in seeds];lines.append(f'- {name}：lazy相对{baseline}误差降低 {sum(d<0 for d in ds)}/{len(ds)}，逐种子差值lazy−baseline={ds}。')
    lines+=['',f'full/lazy共同步前缀检查：{len(pairs)}单元；最大参数差={max([p["max_theta_difference"] for p in pairs],default=0):.6g}；共同记录JVP={sum(p["full_common_jvps"] for p in pairs)} / {sum(p["lazy_common_jvps"] for p in pairs)}。','', '## 参考与限制','', 'NLS使用精确sech(x)exp(it/2)。Gray–Scott沿用E003 32/64空间网格FFT/RK4参考；短窗T=2。CGLE使用64/.002与128/.001积分因子RK4，21时刻比较；不是解析平面波。Allen–Cahn使用原作者AC.mat（512×201），仅原文件单网格参考，尚无本轮独立时空加密。FHN使用官方PDEBench首轨0000（101×128×128×2），配置原始512网格、501时刻、seed0。主评估取每5帧共21时刻全空间点，IC取原始t0帧固定双线性插值并夹到cell中心边缘；主参考仅分发单网格，另外50网格生成器诊断不构成主轨迹加密。数据来源与核查详见data/reference_audit.json；原始白噪声IC未平滑，低容量PINN的IC拟合不足须结合cap和独立审计解释。','', '固定采点和无缩步规则可能迅速自然停滞，不能把1000步上限称为1000步已完成训练；同步数只在实际共同前缀上成立。审计约束越界要区分初态已越界和新增越界，analysis_rows.json逐条保存初态与终点。点残差超限率以每块MSE cap作为逐点平方残差阈值，作为描述指标，不是训练保护规则。','', '线性与直接前向小消融仅在共享初态与Adam末态进行：linear以p×alpha_linear排序并做真实验证；direct按p排序用8个固定二分步长，计入全部small+review真实试算。末态初始不可行的快照明确标记eligible=false，不能纳入可行起点论断。见各单元ablations.json。','', '## 复现','', '```powershell',"Set-Location 'F:/RA/constraint_update_plugin_20261111/e005_evolution_suite'",'& F:/RA/.venv/Scripts/python.exe longrun.py --steps 1000','& F:/RA/.venv/Scripts/python.exe analyze_long.py','```','', '已有.pt时自动复用；要重新测量需将results_long另存后重新运行，保留initial.pt可固定起点。pilot_duplicate_gradients为计时实现修正前试跑，不进入主结果。每条轨迹pt保存全部实际提交theta、初态/终态moments、caps、逐步耗时与候选；JSONL保存所有否决原因，可独立重算。main_table.csv含逐种子主结果；error_trajectories.json含每100提交及预算/末态误差；equivalence.json含full/lazy审计。']
    lines+=['','## 共享快照成本小消融','','下表为同时small和review可行的共享快照一次测量均值，仅selection；公共候选构造未计入此表，主训练端到端时间包含。无重复计时置信区间。对早期仅small筛过的快照，事后补核review，不可行项保留原测量并从本表剔除，详见ablation_audit.json。','','|题目|方法|快照数|selection秒均值|JVP均值|真实复核均值|','|---|---|---:|---:|---:|---:|']
    for name in L.CASES:
        for method in ['tide_full','tide_lazy','linear','direct']:
            rr=[r for r in ablations if r['case']==name and r.get('method')==method and r['eligible']]
            if rr:lines.append(f'|{name}|{method}|{len(rr)}|{np.mean([r["selection_seconds"] for r in rr]):.6g}|{np.mean([r["jvps"] for r in rr]):.3g}|{np.mean([r["checks"] for r in rr]):.3g}|')
    lines+=['','## 初态准备与实际训练总成本','','下表每题已成功启动种子的初始化拟合合计；启动失败成本另列，所有方法共用；训练列为实际轨迹实测端到端合计，含失败试算和日志。自然停滞后没有继续耗时空转，等时间状态维持其终态。数据下载、Python导入、热身、事后独立审计与最终大文件保存不包含在训练列；因此训练列不能视为从零安装至最终报告的全流程成本。','','|题目|初态拟合总秒|Adam训练总秒|slope训练总秒|full训练总秒|lazy训练总秒|','|---|---:|---:|---:|---:|---:|']
    for name in L.CASES:
        rr=[r for r in rows if r['case']==name];prep={r['seed']:r['preparation_seconds'] for r in rr};cost=[sum(r['train_seconds'] for r in rr if r['method']==m) for m in L.METHODS]
        if rr:lines.append('|'+name+'|'+f'{sum(prep.values()):.6g}'+'|'+ '|'.join(f'{x:.6g}' for x in cost)+'|')
    lines+=['','## 启动达标率','']
    for name in L.CASES:
        nn=len(set(r['seed'] for r in rows if r['case']==name));ff=[f for f in failures if f['case']==name];lines.append(f'- {name}：{nn}/5 个种子启动比较，失败种子={[f["seed"] for f in ff]}；失败拟合耗时合计{sum(f["seconds"] for f in ff):.6g}秒。失败不纳入方法均值，也不当作方法成功；完整计入25个尝试单元。')
    lines+=['','来源：[PDEBench官方生成器](https://github.com/pdebench/PDEBench/blob/main/pdebench/data_gen/src/sim_diff_react.py)、[Allen–Cahn原作者数据](https://github.com/maziarraissi/PINNs/blob/master/main/Data/AC.mat)。E004五题单步回放仍应与本报告并列阅读，尤其Helmholtz零收益不由本组时间演化题替代。']
    conclusion='本轮没有验证出可重复的等时间解误差收益。NLS、Gray–Scott、CGLE的保护法误差明显高于1000步加权Adam；Allen–Cahn均值同样较高；FHN虽3/4成功启动种子较Adam误差低，但误差本身很大且独立保护点普遍超限，不能作为达到可比精度的成功证据。full/lazy在24个共同前缀参数完全一致，JVP由214降到141（减少34.1%），证明了本轮决策复现与曲率查询节省；尚未实际实现1000次安全提交。'
    lines.insert(4,conclusion+'\n')
    for f in failures:
        if f.get('timing_source'):lines+=['',f'失败{f["case"]} seed{f["seed"]}耗时{f["seconds"]:.6g}秒为估计，来源：{f["timing_source"]}；原断言失败未保留theta，未补做该种子Adam描述轨迹。']
    (L.ROOT/'连续训练实验报告.md').write_text('\n'.join(lines),encoding='utf-8');L.dump(L.OUT/'analysis_completion.json',{'seconds':time.perf_counter()-start,'rows':len(rows)})
if __name__=='__main__':main()
