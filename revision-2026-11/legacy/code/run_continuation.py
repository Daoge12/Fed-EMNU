"""Pilot-gated post-event retained-only FL; separate canonical provenance."""
import json
import sys
import subprocess
import numpy as np
from pathlib import Path
from run_revision_suite import ROOT,save
from summarize_revision import digest


def audit(suite,pilot):
    rows=[]
    for key,record in suite['runs'].items():
        if record['status']!='complete':raise ValueError('Incomplete continuation')
        d=Path(record['run_dir']);c=json.loads((d/'config.json').read_text());m=json.loads((d/'metrics.json').read_text())
        marker=json.loads((d/'continuation_protocol.json').read_text())
        for name in ('revision_runner.py','revision_evaluation.py','continue_fl_adapter.py'):
            if digest(d/name)!=suite['hashes'][name]:raise ValueError('Source mismatch')
        if marker['protocol_sha256']!=suite['hashes']['protocol'] or marker['rounds']!=(1 if pilot else 5) or marker['quarantine']!=[0] or c['pilot']!=pilot:
            raise ValueError('Continuation protocol mismatch')
        parent=Path(c['checkpoint'])
        if marker['parent_sha256']!=digest(parent) or c['checkpoint_sha256']!=digest(parent):raise ValueError('Wrong parent')
        pred=np.load(d/'test_predictions.npz');p,y=pred['probabilities'],pred['labels']
        pp=np.load(parent.parent/'test_predictions.npz')['probabilities']
        if p.shape!=pp.shape or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('Posterior mismatch')
        for stage,probs in [('before',pp),('after',p)]:
            correct=probs.argmax(1)==y
            for metric,val in [('test_acc',correct.mean()),('forget_acc',correct[y==0].mean()),('retain_acc',correct[y!=0].mean())]:
                if not np.isclose(m[stage][metric],val,atol=1e-12):raise ValueError('Accuracy mismatch')
        splits=json.loads((d/'attack_splits.json').read_text())
        if splits!=json.loads((parent.parent/'attack_splits.json').read_text()):raise ValueError('Attack splits differ')
        retained=json.loads((d/'retained_partitions.json').read_text())
        if retained!=json.loads((parent.parent/'retained_partitions.json').read_text()):raise ValueError('Quarantine differs')
        schedule=json.loads((d/'continuation_schedule.json').read_text())
        if len(schedule)!=(1 if pilot else 5) or any(len(set(x))!=4 for x in schedule):raise ValueError('Wrong new schedule')
        if c.get('reference'):
            ref=Path(c['reference']);rp=np.load(ref.parent/'test_predictions.npz')['probabilities']
            if m['reference_sha256']!=digest(ref) or not np.isclose(m['retrain_distance']['total_variation'],(.5*np.abs(p-rp).sum(1)).mean(),atol=1e-10):raise ValueError('Wrong continued reference')
            if schedule!=json.loads((ref.parent/'continuation_schedule.json').read_text()):raise ValueError('Continued schedules differ')
        rows.append({'key':key,'before_fa':m['before']['forget_acc'],'after_fa':m['after']['forget_acc'],
            'fa_change':m['after']['forget_acc']-m['before']['forget_acc'],
            'before_ra':m['before']['retain_acc'],'after_ra':m['after']['retain_acc'],
            'target_auc':m['membership']['groups']['target']['macro_auc'],
            'tv':m.get('retrain_distance',{}).get('total_variation'),
            'seconds':m['method_seconds'],'metrics_sha256':digest(d/'metrics.json')})
    if len(rows)!=(8 if pilot else 24):raise ValueError('Wrong continuation count')
    return rows


def suite_run(pilot):
    output=ROOT/'experiments'/('continuation-pilot' if pilot else 'continuation')
    output.mkdir(exist_ok=True);path=output/'suite.json'
    hashes={name:digest(ROOT/'code'/name) for name in ('continue_fl_adapter.py','run_continuation.py','revision_runner.py','revision_evaluation.py')}
    hashes['protocol']=digest(ROOT/'experiments/CONTINUATION_PROTOCOL.md')
    state=json.loads(path.read_text()) if path.exists() else {'hashes':hashes,'pilot':pilot,'runs':{}}
    if state['hashes']!=hashes:raise ValueError('Frozen continuation changed')
    state['status']='running';save(path,state)
    core=json.loads((ROOT/'experiments'/('request-sequence-pilot' if pilot else 'core-suite')/'suite.json').read_text())
    for setting in ('iid','noniid'):
        for seed in ([42] if pilot else [42,43,44]):
            prefix=f'{setting}_seed{seed}';reference=None
            for method in ('retrain','ft','noise','unsir'):
                key=prefix+'_'+method
                if pilot:parent_key=prefix+('_reference0' if method=='retrain' else '_'+method+'_first')
                else:parent_key=key
                parent=Path(core['runs'][parent_key]['run_dir'])/'final_model.pt'
                old=state['runs'].get(key,{})
                if old.get('status')=='complete':
                    if method=='retrain':reference=Path(old['run_dir'])/'final_model.pt'
                    continue
                command=[sys.executable,'-u',str(ROOT/'code/continue_fl_adapter.py'),'--mode',method,
                    '--checkpoint',str(parent),'--output_dir',str(output/key),'--data_root',str(ROOT/'code/data'),
                    '--seed',str(seed),'--split_seed',str(seed),'--unlearn_seed',str(seed),'--retrain_seed',str(seed+1000)]
                if setting=='iid':command+=['--iid']
                if pilot:command+=['--pilot','--global_epochs','1','--local_epochs','1','--limit_train','1000','--limit_test','500']
                if reference:command+=['--reference',str(reference)]
                state['runs'][key]={'status':'running','command':command};save(path,state)
                print('START',key,flush=True)
                with (output/'suite.log').open('a',encoding='utf-8') as log:
                    code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
                statuses=sorted((output/key).glob('*/status.json'))
                if code or not statuses or json.loads(statuses[-1].read_text())['status']!='complete':
                    state['status']='failed';state['runs'][key]['status']='failed';save(path,state)
                    raise RuntimeError('Continuation failed '+key)
                folder=statuses[-1].parent
                state['runs'][key].update(status='complete',run_dir=str(folder));save(path,state)
                if method=='retrain':reference=folder/'final_model.pt'
    rows=audit(state,pilot)
    save(output/'audit.json',{'complete':True,'pilot':pilot,'rows':rows})
    state['status']='complete';save(path,state)
    print('AUDIT_COMPLETE',output,flush=True)

if __name__=='__main__':
    suite_run(True)
    suite_run(False)
