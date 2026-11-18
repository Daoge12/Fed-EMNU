"""Paired, restartable second-round controls. No retraining or test-based tuning."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r'C:\Users\le\Documents\Codex\2026-11-09\ni\outputs\Fed-EMNU-revision')
VARIANTS = {
    'fed_high': ('noise', []),
    'unsir_high': ('unsir', []),
    'fed_low': ('noise', ['--noise_epochs','5','--repair_epochs','1']),
    'unsir_low': ('unsir', ['--noise_epochs','5','--repair_epochs','1']),
    'gaussian': ('noise', ['--noise_epochs','0']),
    'equal_event': ('noise', ['--event_aggregation','equal']),
}

def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this timing study')
    manifest = json.loads((SOURCE/'experiments/core-suite/suite.json').read_text())['runs']
    extra = json.loads((SOURCE/'experiments/core-extra-seeds/suite.json').read_text())['runs']
    manifest.update(extra)
    for setting in ('iid','noniid'):
        for seed in (42,43,44,45):
            for mode in ('pretrain','retrain'):
                path = Path(manifest[f'{setting}_seed{seed}_{mode}']['run_dir'])/'final_model.pt'
                assert path.is_file(), path
    assert (SOURCE/'code/data/cifar-10-batches-py/data_batch_1').is_file()
    print('PREFLIGHT_OK',torch.__version__,torch.cuda.get_device_name(0),flush=True)
    if args.check_only:
        return
    out = ROOT/'experiments/round2'
    out.mkdir(parents=True,exist_ok=True)
    state_path = out/'suite.json'
    hashes = {n:hashlib.sha256((ROOT/'code'/n).read_bytes()).hexdigest()
              for n in ('revision_runner.py','revision_evaluation.py','run_round2.py')}
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'source_hashes':hashes,'status':'running','runs':{},
        'started_utc':datetime.now(timezone.utc).isoformat(),
        'protocol':'experiments/ROUND2_PROTOCOL.md'}
    assert state['source_hashes']==hashes, 'Source changed: use a fresh suite'
    tasks = [(setting,seed,name,mode,flags,'random',0)
             for setting in ('iid','noniid') for seed in (42,43,44)
             for name,(mode,flags) in VARIANTS.items()]
    # Repeat original Non-IID recipe on both a normal and an outlier checkpoint.
    tasks += [('noniid',seed,'timing_repeat','noise',[],'confidence',rep)
              for seed in (42,45) for rep in (1,2,3)]
    for setting,seed,name,mode,flags,selection,rep in tasks:
        key=f'{setting}_seed{seed}_{name}_{rep}'
        previous=state['runs'].get(key,{})
        if previous.get('status')=='complete':
            assert (Path(previous['run_dir'])/'metrics.json').is_file()
            continue
        checkpoint=Path(manifest[f'{setting}_seed{seed}_pretrain']['run_dir'])/'final_model.pt'
        reference=Path(manifest[f'{setting}_seed{seed}_retrain']['run_dir'])/'final_model.pt'
        command=[sys.executable,'-u',str(ROOT/'code/revision_runner.py'),'--mode',mode,
            '--data_root',str(SOURCE/'code/data'),'--output_dir',str(out/key),
            '--checkpoint',str(checkpoint),'--reference',str(reference),
            '--seed',str(seed),'--split_seed',str(seed),'--unlearn_seed',str(seed),
            '--retrain_seed',str(1000+seed),'--selection',selection,*flags]
        if setting=='iid': command+=['--iid']
        record={'command':command,'status':'running','setting':setting,'seed':seed,
                'variant':name,'repeat':rep,'started_utc':datetime.now(timezone.utc).isoformat()}
        state['runs'][key]=record
        save(state_path,state)
        print('START',key,flush=True)
        with (out/(key+'.log')).open('w',encoding='utf-8') as log:
            proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT/'code')
        candidates=sorted((out/key).glob('*/metrics.json'))
        if proc.returncode or not candidates:
            record.update(status='failed',exit_code=proc.returncode)
            state['status']='failed';save(state_path,state)
            raise RuntimeError(f'{key} failed; see {out/(key+".log")}')
        record.update(status='complete',run_dir=str(candidates[-1].parent),
                      finished_utc=datetime.now(timezone.utc).isoformat())
        save(state_path,state)
        print('COMPLETE',key,flush=True)
    state['status']='complete'
    state['finished_utc']=datetime.now(timezone.utc).isoformat()
    save(state_path,state)
    print('ROUND2_COMPLETE',flush=True)

if __name__=='__main__': main()
