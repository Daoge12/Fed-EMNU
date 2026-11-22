"""Summarize complete audited routes, including total cost of sequential paths."""
import json
from pathlib import Path
from run_request_sequence import audit
from summarize_revision import ROOT,stats,digest


def main():
    path=ROOT/'experiments/request-sequence/suite.json'
    suite=json.loads(path.read_text())
    if suite['status']!='complete' or suite['spec']['pilot']:
        raise ValueError('Complete formal request sequence required')
    events=audit(suite,False)
    core=json.loads((ROOT/'experiments/core-suite/suite.json').read_text())
    groups={}; rows=[]
    for event in events:
        setting,rest=event['key'].split('_seed')
        seed,method,route=rest.split('_')
        row=dict(event,setting=setting,seed=int(seed),method=method,route=route)
        folder=Path(suite['runs'][event['key']]['run_dir'])
        metrics=json.loads((folder/'metrics.json').read_text())
        c=metrics['communication']
        row['total_seconds']=metrics['method_seconds']
        row['total_steps']=metrics['optimizer_steps']
        row['total_payload_bytes']=sum(c.get(k,0) for k in ('model_download_bytes','model_upload_bytes','noise_download_bytes'))
        if route=='sequential':
            first=Path(core['runs'][f'{setting}_seed{seed}_{method}']['run_dir'])
            m=json.loads((first/'metrics.json').read_text()); c=m['communication']
            row['total_seconds']+=m['method_seconds']
            row['total_steps']+=m['optimizer_steps']
            row['total_payload_bytes']+=sum(c.get(k,0) for k in ('model_download_bytes','model_upload_bytes','noise_download_bytes'))
            if metrics['membership']['groups']['target']['class_count']!=2 or metrics['membership']['groups']['retained']['class_count']!=8:
                raise ValueError('Incorrect cumulative class groups')
        group=groups.setdefault(setting+'/'+method+'/'+route,{})
        for key in ('fa','ra','target_auc','tv','class0_change','total_seconds','total_steps','total_payload_bytes'):
            group.setdefault(key,[]).append(row[key])
        rows.append(row)
    output=path.parent/'summary.json'
    output.write_text(json.dumps({'suite_sha256':digest(path),'rows':rows,
        'summary':{g:{m:stats(v) for m,v in metrics.items()} for g,metrics in groups.items()},
        'interpretation':'Sequential costs include both events; different total budgets. Class0 change is event2 parent-to-final rebound; not a privacy certificate.'},indent=2),encoding='utf-8')
    print(output)


if __name__=='__main__':
    main()
