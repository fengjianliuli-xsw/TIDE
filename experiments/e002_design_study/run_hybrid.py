import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
for tag,extra in [('hybrid',[]),('hybrid_time',['--freqs','1','--time-budget','3'])]:
    with (root/f'{tag}.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,str(root/'study.py'),'--tag',tag,'--steps','200' if tag=='hybrid' else '5000','--methods','gn_bc_tangent',*extra],stdout=log,stderr=subprocess.STDOUT,check=True)
with (root/'analysis.log').open('w',encoding='utf-8') as log:
    subprocess.run([sys.executable,str(root/'summarize.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,str(root/'report.py')],check=True)
