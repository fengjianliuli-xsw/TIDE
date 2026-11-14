import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
with (root/'time_budget.log').open('w',encoding='utf-8') as log:
    subprocess.run([sys.executable,str(root/'study.py'),'--tag','time_budget','--steps','5000','--freqs','1','--methods','fo_all_tangent,gn_tangent,oracle_tangent,gn_cont_tangent','--time-budget','3'],stdout=log,stderr=subprocess.STDOUT,check=True)
with (root/'analysis.log').open('w',encoding='utf-8') as log:
    subprocess.run([sys.executable,str(root/'summarize.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
print('Time-budget experiments and aggregation complete.')
