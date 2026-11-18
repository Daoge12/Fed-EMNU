"""Execute counterbalanced wall-clock caps without selecting by test outcomes."""
import json
import sys
import hashlib
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(r'C:\Users\le\Documents\Codex\2026-11-09\ni\outputs\Fed-EMNU-revision')
OUT=ROOT/'experiments/walltime'
def save(p,d):p.write_text(json.dumps(d,indent=2),encoding='utf-8')
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    hashes={n:hashlib.sha256((ROOT/'code'/n).read_bytes()).hexdigest() for n in ['walltime_adapter.py','run_walltime.py','revision_runner.py','revision_evaluation.py']}
    path=OUT/'suite.json'
    state=json.loads(path.read_text()) if path.exists() else {'hashes':hashes,'status':'running','runs':{}}
    assert state['hashes']==hashes
    old=json.loads((SOURCE/'experiments/core-suite/suite.json').read_text())['runs']
    for setting in ['iid','noniid']:
        for seed in [42,43,44]:
            for budget in [10,20,30,40]:
                modes=['noise','unsir'] if (seed+budget//10)%2==0 else ['unsir','noise']
                for mode in modes:
                    key=f'{setting}_seed{seed}_{budget}s_{mode}'
                    if state['runs'].get(key,{}).get('status')=='complete':continue
                    cmd=[sys.executable,'-u',str(ROOT/'code/walltime_adapter.py'),'--wall_budget',str(budget),
                         '--mode',mode,'--data_root',str(SOURCE/'code/data'),'--output_dir',str(OUT/key),
                         '--checkpoint',str(Path(old[f'{setting}_seed{seed}_pretrain']['run_dir'])/'final_model.pt'),
                         '--reference',str(Path(old[f'{setting}_seed{seed}_retrain']['run_dir'])/'final_model.pt'),
                         '--seed',str(seed),'--split_seed',str(seed),'--unlearn_seed',str(seed),'--retrain_seed',str(seed+1000),
                         '--selection','random']
                    if setting=='iid':cmd+=['--iid']
                    if mode=='unsir':cmd+=['--noise_epochs','5','--repair_epochs','1']
                    rec={'status':'running','command':cmd,'setting':setting,'seed':seed,'budget':budget,'mode':mode}
                    state['runs'][key]=rec;save(path,state);print('START',key,flush=True)
                    with (OUT/(key+'.log')).open('w',encoding='utf-8') as f:proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
                    dirs=sorted((OUT/key).glob('*/metrics.json'))
                    if proc.returncode or not dirs:
                        rec['status']='failed';state['status']='failed';save(path,state);raise RuntimeError(key)
                    rec.update(status='complete',run_dir=str(dirs[-1].parent));save(path,state);print('COMPLETE',key,flush=True)
    state['status']='complete';save(path,state)
if __name__=='__main__':main()
