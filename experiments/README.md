# PINN 约束感知更新选择：首轮机制实验

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
