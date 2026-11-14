import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
commands=[
 ['--tag','replay','--replay','--methods',''],
 ['--tag','continuous','--methods','gn_cont_tangent,pair_oracle_raw','--steps','200'],
 ['--tag','sgdm','--optimizer','sgdm','--freqs','1','--methods','oracle_tangent,gn_tangent,fo_all_tangent,gn_repair_raw','--steps','200'],
]
for cmd in commands:
    tag=cmd[1]
    print('Starting',tag,flush=True)
    with (root/f'{tag}.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,str(root/'study.py'),*cmd],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Completed',tag,flush=True)
with (root/'analysis.log').open('w',encoding='utf-8') as log:
    subprocess.run([sys.executable,str(root/'summarize.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
print('All followups and aggregation completed',flush=True)
