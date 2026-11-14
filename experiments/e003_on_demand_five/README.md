# E003 按需边界/初值曲率：五问题先导

阅读实验报告.md和实验记录.md；main、budget各45条轨迹，cache_main、cache_budget各15条，pilot15条。

复现（新tag避免覆盖）：
```
& F:/RA/.venv/Scripts/python.exe F:/RA/constraint_update_plugin_20261111/e003_on_demand_five/bench.py --tag reproduction --steps 120
```

正式配置快照见main/source.py与budget/source.py；report.py重建本轮报告和图。使用现有CPU PyTorch、NumPy、Matplotlib，无新增安装。Gray参考保存在gray_reference.npz，生成和加密检查记录在reference_check.json。首轮使用相对起点保护阈值，不等价于固定物理精度或完整求解。
