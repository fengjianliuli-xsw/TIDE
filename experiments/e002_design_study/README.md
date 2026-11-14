# E002: 方向选择与约束预算设计研究

先读实验报告.md，再读实验记录.md。实验是探索性，111条正式/后续轨迹，加8条先导。

环境复用 F:/RA/.venv/Scripts/python.exe，CPU PyTorch，NumPy，Matplotlib；所有数据在本目录。

复现时使用新tag，例如：
```
& F:/RA/.venv/Scripts/python.exe F:/RA/constraint_update_plugin_20261111/e002_design_study/study.py --tag reproduction --steps 200 --methods oracle_tangent,fo_all_tangent,gn_tangent
```

完整原主实验方法列表见main/config.json，不能使用当前脚本默认方法列表来冒充原主实验（后续已新增方法）。每个子目录source.py可核对当次精确源码。summarize.py和report.py重建已有数据的汇总与报告；运行study.py不覆盖已存在tag。
