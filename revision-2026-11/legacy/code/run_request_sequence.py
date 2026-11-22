"""Serial pilot-gated multi-target/sequential request experiment."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from run_revision_suite import ROOT,save
from summarize_revision import digest


def audit(state,pilot):
    evidence=[]
    for key,run in state['runs'].items():
        if run['status']!='complete':
            raise ValueError('Incomplete request sequence')
        folder=Path(run['run_dir'])
        config=json.loads((folder/'config.json').read_text())
        metrics=json.loads((folder/'metrics.json').read_text())
        if bool(config['pilot'])!=pilot or metrics['pilot']!=pilot:
            raise ValueError('Pilot marker mismatch')
        pz=np.load(folder/'test_predictions.npz'); p,y=pz['probabilities'],pz['labels']
        if p.shape!=((500 if pilot else 10000),10) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5):
            raise ValueError('Invalid sequence predictions')
        target=np.isin(y,config['unlearn_class']); correct=p.argmax(1)==y
        for metric,actual in [('test_acc',correct.mean()),('forget_acc',correct[target].mean()),('retain_acc',correct[~target].mean())]:
            if not np.isclose(metrics['after'][metric],actual,atol=1e-12):
                raise ValueError('Accuracy mismatch')
        splits=json.loads((folder/'attack_splits.json').read_text())
        if len(splits)!=10:
            raise ValueError('Missing attack classes')
        for split in splits.values():
            for group in ('member','nonmember'):
                if set(split[group+'_calibration']) & set(split[group+'_test']):
                    raise ValueError('Attack calibration leakage')
        for name in ('revision_runner.py','revision_evaluation.py'):
            if digest(folder/name)!=state['spec']['hashes'][name]:
                raise ValueError('Frozen runner mismatch')
        if 'sequence_adapter_sha256' not in config:
            continue
        marker=json.loads((folder/'sequence_protocol.json').read_text())
        for name,field in [('request_sequence_adapter.py','sequence_adapter_sha256'),('REQUEST_SEQUENCE_PROTOCOL.md','request_sequence_protocol_sha256')]:
            if digest(folder/name)!=config[field] or config[field]!=state['spec']['hashes'][name]:
                raise ValueError('Sequence source mismatch')
        parent=Path(config['checkpoint']); reference=Path(config['reference'])
        if marker['parent_sha256']!=digest(parent) or config['checkpoint_sha256']!=digest(parent) or marker['reference_sha256']!=digest(reference) or metrics['reference_sha256']!=digest(reference):
            raise ValueError('Wrong parent/reference')
        pp=np.load(parent.parent/'test_predictions.npz')['probabilities']
        rp=np.load(reference.parent/'test_predictions.npz')['probabilities']
        for metric,actual in [('test_acc',(pp.argmax(1)==y).mean()),('forget_acc',(pp.argmax(1)[target]==y[target]).mean()),('retain_acc',(pp.argmax(1)[~target]==y[~target]).mean())]:
            if not np.isclose(metrics['before'][metric],actual,atol=1e-12):
                raise ValueError('Before metrics do not match actual parent')
        if not np.isclose(metrics['retrain_distance']['total_variation'],(.5*np.abs(p-rp).sum(1)).mean(),atol=1e-10):
            raise ValueError('Reference TV mismatch')
        for other in (parent.parent,reference.parent):
            if json.loads((other/'partitions.json').read_text())!=json.loads((folder/'partitions.json').read_text()) or json.loads((other/'attack_splits.json').read_text())!=splits:
                raise ValueError('Partition/attack splits differ across paths')
        indices=json.loads((folder/'data_indices.json').read_text())['train']
        # CIFAR-10 full labels and pilot selected order are both explicit.
        import pickle
        full=[]
        for i in range(1,6):
            with (ROOT/f'code/data/cifar-10-batches-py/data_batch_{i}').open('rb') as stream:
                full.extend(pickle.load(stream,encoding='bytes')[b'labels'])
        labels=np.asarray(full)[indices]
        retain=json.loads((folder/'retained_partitions.json').read_text())
        parts=json.loads((folder/'partitions.json').read_text())
        expected=[[i for i in part if labels[i] not in config['unlearn_class']] for part in parts]
        if retain!=expected:
            raise ValueError('Cumulative quarantine mismatch')
        selection=json.loads((folder/'selection.json').read_text())
        if any(not set(record['indices'])<=set(retain[int(cid)]) for cid,record in selection.items()):
            raise ValueError('Selection includes inaccessible/cumulative deleted examples')
        class_fa={str(c):float(correct[y==c].mean()) for c in (0,1)}
        evidence.append({'key':key,'class_accuracy':class_fa,
            'class0_before':float((pp.argmax(1)[y==0]==y[y==0]).mean()),
            'class0_change':class_fa['0']-float((pp.argmax(1)[y==0]==y[y==0]).mean()),
            'fa':metrics['after']['forget_acc'],'ra':metrics['after']['retain_acc'],
            'target_auc':metrics['membership']['groups']['target']['macro_auc'],
            'tv':metrics['retrain_distance']['total_variation'],
            'seconds':metrics['method_seconds'],'steps':metrics['optimizer_steps'],
            'metrics_sha256':digest(folder/'metrics.json')})
    return evidence


def run_suite(pilot):
    output=ROOT/('experiments/request-sequence-pilot' if pilot else 'experiments/request-sequence')
    output.mkdir(exist_ok=True)
    hashes={name:digest(ROOT/'code'/name) for name in ('revision_runner.py','revision_evaluation.py','request_sequence_adapter.py','run_request_sequence.py')}
    hashes['REQUEST_SEQUENCE_PROTOCOL.md']=digest(ROOT/'experiments/REQUEST_SEQUENCE_PROTOCOL.md')
    spec={'pilot':pilot,'hashes':hashes,'seeds':[42] if pilot else [42,43,44],'settings':['iid','noniid']}
    path=output/'suite.json'
    state=json.loads(path.read_text()) if path.exists() else {'spec':spec,'runs':{}}
    if state['spec']!=spec:
        raise ValueError('Sequence source/protocol changed; use new experiment version')
    state['status']='running';save(path,state)
    core=json.loads((ROOT/'experiments/core-suite/suite.json').read_text())
    def run(key,mode,common,extra,adapter=False):
        old=state['runs'].get(key,{})
        if old.get('status')=='complete':
            return Path(old['run_dir'])/'final_model.pt'
        command=[sys.executable,'-u',str(ROOT/'code'/('request_sequence_adapter.py' if adapter else 'revision_runner.py')),
                 '--mode',mode,'--output_dir',str(output/key),'--data_root',str(ROOT/'code/data')]+common+extra
        state['runs'][key]={'status':'running','command':command,'started':datetime.now(timezone.utc).isoformat()};save(path,state)
        print('START',key,flush=True)
        with (output/'suite.log').open('a',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        statuses=sorted((output/key).glob('*/status.json'))
        if result.returncode or not statuses or json.loads(statuses[-1].read_text())['status']!='complete':
            state['runs'][key]['status']='failed';state['status']='failed';save(path,state)
            raise RuntimeError('Sequence stage failed '+key)
        folder=statuses[-1].parent
        state['runs'][key].update(status='complete',run_dir=str(folder));save(path,state)
        return folder/'final_model.pt'
    for setting in spec['settings']:
        for seed in spec['seeds']:
            prefix=f'{setting}_seed{seed}'
            common=['--seed',str(seed),'--split_seed',str(seed),'--unlearn_seed',str(seed),'--retrain_seed',str(seed+1000)]
            if setting=='iid': common+=['--iid']
            if pilot: common+=['--pilot','--global_epochs','1','--local_epochs','1','--limit_train','1000','--limit_test','500']
            if pilot:
                original=run(prefix+'_pretrain','pretrain',common,[])
                ref0=run(prefix+'_reference0','retrain',common,['--checkpoint',str(original)])
            else: original=Path(core['runs'][prefix+'_pretrain']['run_dir'])/'final_model.pt'
            ref01=run(prefix+'_reference01','retrain',common,['--checkpoint',str(original),'--unlearn_class','0','1'])
            for method in ('ft','ga','noise','unsir'):
                options=['--noise_epochs','5','--repair_epochs','1'] if method=='unsir' else []
                if pilot:
                    first=run(prefix+'_'+method+'_first',method,common,['--checkpoint',str(original),'--reference',str(ref0)]+options)
                else: first=Path(core['runs'][prefix+'_'+method]['run_dir'])/'final_model.pt'
                for route,parent,current,prior in [('simultaneous',original,['0','1'],[]),('sequential',first,['1'],['0'])]:
                    extra=['--checkpoint',str(parent),'--reference',str(ref01),'--unlearn_class','0','1','--current_targets']+current
                    if prior: extra+=['--prior_targets']+prior
                    run(prefix+'_'+method+'_'+route,method,common,extra+options,True)
    evidence=audit(state,pilot)
    if len(state['runs'])!=(30 if pilot else 54) or len(evidence)!=(16 if pilot else 48):
        raise ValueError('Incomplete expected sequence matrix')
    save(output/'audit.json',{'pilot':pilot,'complete':True,'events':evidence})
    state['status']='complete';save(path,state)
    print('AUDIT_COMPLETE',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot-only',action='store_true')
    args=parser.parse_args()
    run_suite(True)
    if not args.pilot_only:
        run_suite(False)
