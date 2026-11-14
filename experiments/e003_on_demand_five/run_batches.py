import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
for tag,args in [('main',['--steps','120']),('budget',['--steps','5000','--budget','3'])]:
    print('Starting',tag,flush=True)
    with (root/f'{tag}.log').open('w',encoding='utf-8') as f:
        subprocess.run([sys.executable,str(root/'bench.py'),'--tag',tag,*args],stdout=f,stderr=subprocess.STDOUT,check=True)
    print('Completed',tag,flush=True)
