"""Pilot-gated serial twenty-client extension, restartable by suite manifests."""
import json
import pickle
import subprocess
import sys
from pathlib import Path
import numpy as np
from run_revision_suite import ROOT,save
from summarize_revision import digest,read_runs


def audit(suite,pilot):
    expected={f'{setting}_seed{seed}_{mode}' for setting in ('iid','noniid')
              for seed in ([42] if pilot else [42,43,44])
              for mode in ('pretrain','retrain','ft','ga','noise','unsir')}
    if suite['status']!='complete' or set(suite['runs'])!=expected:
        raise ValueError('Complete twenty-client matrix required')
    full=[]
    for i in range(1,6):
        with (ROOT/f'code/data/cifar-10-batches-py/data_batch_{i}').open('rb') as stream:
            full.extend(pickle.load(stream,encoding='bytes')[b'labels'])
    for key,run in suite['runs'].items():
        folder=Path(run['run_dir'])
        c=json.loads((folder/'config.json').read_text()); m=json.loads((folder/'metrics.json').read_text())
        if c['num_clients']!=20 or bool(c['pilot'])!=pilot or bool(m['pilot'])!=pilot or run['status']!='complete':
            raise ValueError('Wrong twenty-client marker')
        for name,h in suite['spec']['source_hashes'].items():
            if digest(folder/name)!=h: raise ValueError('Source mismatch')
        parts=json.loads((folder/'partitions.json').read_text())
        retain=json.loads((folder/'retained_partitions.json').read_text())
        ids=json.loads((folder/'data_indices.json').read_text())['train']
        labels=np.asarray(full)[ids]
        if len(parts)!=20 or any(not p for p in parts) or sorted(i for p in parts for i in p)!=list(range(len(ids))):
            raise ValueError('Invalid twenty-client partition')
        if retain!=[[i for i in part if labels[i]!=0] for part in parts]:
            raise ValueError('Wrong retained partition')
        schedule=json.loads((folder/'client_schedule.json').read_text())
        if len(schedule)!=(1 if pilot else 50) or any(len(set(p))!=8 or any(i<0 or i>=20 for i in p) for p in schedule):
            raise ValueError('Wrong participation schedule')
        pred=np.load(folder/'test_predictions.npz'); p,y=pred['probabilities'],pred['labels']
        if p.shape!=((500 if pilot else 10000),10) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5):
            raise ValueError('Invalid posterior')
        correct=p.argmax(1)==y
        for metric,v in [('test_acc',correct.mean()),('forget_acc',correct[y==0].mean()),('retain_acc',correct[y!=0].mean())]:
            if not np.isclose(m['after'][metric],v,atol=1e-12): raise ValueError('Accuracy mismatch')
        splits=json.loads((folder/'attack_splits.json').read_text())
        if set(splits)!={str(i) for i in range(10)}: raise ValueError('Missing attack class')
        for cls,split in splits.items():
            for group,targets in [('member',labels),('nonmember',y)]:
                a,b=split[group+'_calibration'],split[group+'_test']
                if set(a)&set(b) or len(set(a+b))!=len(a+b) or any(targets[i]!=int(cls) for i in a+b):
                    raise ValueError('Attack split mismatch')
        prefix=key.rsplit('_',1)[0]
        original=Path(suite['runs'][prefix+'_pretrain']['run_dir'])
        if c['mode']!='pretrain':
            if c['checkpoint_sha256']!=digest(original/'final_model.pt') or m['before']!=json.loads((original/'metrics.json').read_text())['after']:
                raise ValueError('Wrong common starting model')
        if c['mode'] not in ('pretrain','retrain'):
            ref=Path(suite['runs'][prefix+'_retrain']['run_dir'])
            rp=np.load(ref/'test_predictions.npz')['probabilities']
            if m['reference_sha256']!=digest(ref/'final_model.pt') or not np.isclose(m['retrain_distance']['total_variation'],(.5*np.abs(p-rp).sum(1)).mean(),atol=1e-10):
                raise ValueError('Wrong reference')
            affected=[cid for cid,part in enumerate(parts) if any(labels[i]==0 for i in part)]
            if m['communication']['participating_clients']!=affected:
                raise ValueError('Wrong target-holder participation')
            if json.loads((ref/'attack_splits.json').read_text())!=splits:
                raise ValueError('Attack splits differ')
    if not pilot:
        _,_,missing=read_runs(suite)
        if missing: raise ValueError('Canonical audit incomplete')


def main():
    state_path=ROOT/'experiments/clients20-queue.json'
    hashes={name:digest(ROOT/'code'/name) for name in ('run_clients20.py','run_revision_suite.py','revision_runner.py','revision_evaluation.py')}
    hashes['protocol']=digest(ROOT/'experiments/CLIENTS20_PROTOCOL.md')
    state=json.loads(state_path.read_text()) if state_path.exists() else {'hashes':hashes}
    if state['hashes']!=hashes: raise ValueError('Frozen coordinator/protocol changed')
    try:
        for pilot,name in [(True,'clients20-pilot'),(False,'extension-clients20')]:
            state.update(status='running_pilot' if pilot else 'running_full');save(state_path,state)
            command=[sys.executable,'-u',str(ROOT/'code/run_revision_suite.py'),'--num-clients','20','--output','experiments/'+name]
            if pilot: command+=['--pilot','--seeds','42']
            with (ROOT/'experiments/clients20-queue.log').open('a',encoding='utf-8') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            suite=json.loads((ROOT/'experiments'/name/'suite.json').read_text())
            audit(suite,pilot)
            save(ROOT/'experiments'/name/'clients20-audit.json',{'complete':True,'pilot':pilot,'runs':len(suite['runs']),'suite_sha256':digest(ROOT/'experiments'/name/'suite.json'),'hashes':hashes})
            print('AUDIT_COMPLETE',name,flush=True)
        subprocess.run([sys.executable,str(ROOT/'code/summarize_revision.py'),'--suite','experiments/extension-clients20/suite.json'],cwd=ROOT,check=True)
        state['status']='complete';save(state_path,state)
    except Exception as error:
        state.update(status='failed',error=repr(error));save(state_path,state)
        raise


if __name__=='__main__': main()
