"""Reaudit and integrate retained-only continuation, without PDF generation."""
import json
from pathlib import Path
import numpy as np
from run_continuation import audit
from summarize_revision import ROOT,stats,digest


def main():
    path=ROOT/'experiments/continuation/suite.json'
    suite=json.loads(path.read_text())
    if suite['status']!='complete' or suite['pilot']:
        raise ValueError('Complete scientific continuation required')
    rows=audit(suite,False)
    for row in rows:
        d=Path(suite['runs'][row['key']]['run_dir'])
        config=json.loads((d/'config.json').read_text())
        metrics=json.loads((d/'metrics.json').read_text())
        parent=Path(config['checkpoint']).parent
        original=json.loads((parent/'metrics.json').read_text())
        # Compare like-for-like attack splits/calibration, not utility alone.
        row['before_auc']=original['membership']['groups']['target']['macro_auc']
        row['auc_change']=row['target_auc']-row['before_auc']
        row['ra_change']=row['after_ra']-row['before_ra']
        row['retained_auc']=metrics['membership']['groups']['retained']['macro_auc']
        if config.get('reference'):
            row['delta_reference_auc']=metrics['membership_difference_from_retrain']['target']['macro_auc']
            row['prediction_disagreement']=metrics['retrain_distance']['prediction_disagreement']
        else:
            row['delta_reference_auc']=0.0
            row['prediction_disagreement']=0.0
            row['tv']=0.0
        for attack_name in ('membership','reference_membership'):
            if attack_name not in metrics:continue
            attack=metrics[attack_name]
            if set(attack['per_class'])!={str(c) for c in range(10)}:
                raise ValueError('Incomplete class attack')
            for group in ('target','retained'):
                records=[v for c,v in attack['per_class'].items() if (int(c)==0)==(group=='target')]
                for key in ('auc','advantage'):
                    if not np.isclose(np.mean([v[key] for v in records]),attack['groups'][group]['macro_'+key],atol=1e-12):
                        raise ValueError('Attack macro mismatch')
    groups={}
    for row in rows:
        setting,rest=row['key'].split('_seed'); seed,method=rest.split('_')
        row.update(setting=setting,seed=int(seed),method=method)
        group=groups.setdefault(setting+'/'+method,{})
        for key in ('before_fa','after_fa','fa_change','before_ra','after_ra','ra_change',
                    'before_auc','target_auc','auc_change','retained_auc','delta_reference_auc',
                    'tv','prediction_disagreement','seconds'):
            group.setdefault(key,[]).append(row[key])
    summary={g:{m:stats(v) for m,v in ms.items()} for g,ms in groups.items()}
    output=path.parent/'summary.json'
    output.write_text(json.dumps({'suite_sha256':digest(path),'complete':True,'rows':rows,
        'summary':summary,'limitations':'Five retained-only rounds; one class, one architecture, ten clients, n3. No reinsertion arm or all-future guarantee. Reference continues original retained state; not a newly initialized reference.'},indent=2),encoding='utf-8')
    def pm(g,m,scale=1,d=2):
        e=g[m];return f"${e['mean']*scale:.{d}f} \\pm {e['sd']*scale:.{d}f}$"
    lines=[r'\subsection{Retained-Only Federated Training after Revocation}',r'\label{sec:continuation}',
        'To test subsequent behavioral recovery, each class-0 output of Fed-EMNU, '
        'UNSIR-style and Finetune receives five additional federated rounds using '
        'only retained data. The independently trained retained-data reference '
        'continues from its own saved state with the same additional budget. '
        'This continued reference is not a newly initialized retraining run. '
        'CIFAR-10/ResNet18, ten clients, IID and Dirichlet $\\alpha=0.5$, seeds '
        '42--44 are used. All 24 formal continuations completed and passed the '
        'artifact audit; eight reduced-data pilot continuations are excluded.',
        'Class 0 remains quarantined at every client throughout continuation. '
        'Each new round selects four clients with a common seed-specific schedule '
        'generated from seed $+70000$, and applies five local epochs with fresh '
        'Adam at 0.01 and batch size 128. Local shuffling uses seed $+50000+'
        '10t+i$ for zero-based round $t$ and client $i$. The adapter explicitly '
        'loads each parent state; reference metadata retains the evaluator\'s '
        'retrain mode only for compatibility and is accompanied by a continuation '
        'marker stating the changed semantics. Original schedules, source and '
        'parent hashes remain saved. Parent and final accuracy are recomputed '
        'from predictions; attacks share historical-member splits and use the '
        'same declared calibration procedure. TV is against the continued reference. '
        'Values are mean $\\pm$ sample SD, $n=3$.']
    for setting in ('iid','noniid'):
        lines += [r'\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{3pt}',
            '\\caption{'+('IID' if setting=='iid' else 'Non-IID')+
            ' retained-only continuation. FA/RA changes are final minus parent, in percentage points; all other columns are final values.}',
            '\\label{tab:continuation-'+setting+'}',
            r'\begin{tabular}{lrrrrrr}\toprule Parent & FA (\%) & RA (\%) & $\Delta$FA (pp) & $\Delta$RA (pp) & Target AUC & TV\\\midrule']
        for method,name in [('retrain','Retained reference'),('ft','Finetune'),('noise','Fed-EMNU'),('unsir','UNSIR-style')]:
            g=summary[setting+'/'+method]
            lines.append(name+' & '+' & '.join([pm(g,'after_fa',100),pm(g,'after_ra',100),
                pm(g,'fa_change',100),pm(g,'ra_change',100),pm(g,'target_auc',d=3),
                '0 (self)' if method=='retrain' else pm(g,'tv',d=3)])+r'\\')
        lines += [r'\bottomrule\end{tabular}\end{table}']
        g=summary[setting+'/noise']; ref=summary[setting+'/retrain']
        lines.append(('Under IID' if setting=='iid' else 'Under Non-IID')+
            ', Fed-EMNU target AUC changes from '+pm(g,'before_auc',d=3)+' to '+
            pm(g,'target_auc',d=3)+', while the continued reference has AUC '+pm(ref,'target_auc',d=3)+
            '. The final paired AUC difference from the continued reference is '+pm(g,'delta_reference_auc',d=3)+
            '. Continued training is additional compute: Fed-EMNU continuation takes '+pm(g,'seconds')+
            ' s, excluding the original unlearning event, evaluation and real transport. '
            'The continuation wrapper does not measure transport payload; its empty '
            'communication record must not be interpreted as zero network traffic.')
    increases=[r for r in rows if r['method']=='noise' and r['fa_change']>1e-12]
    lines.append(f'Class-0 FA increases in {len(increases)} of the six Fed-EMNU '
        'continuations. This is a finite-horizon behavioral check under strict '
        'quarantine, not evidence of equality to retraining or a privacy guarantee. '
        'In particular, suppressed target accuracy can coexist with membership '
        'scores different from the continued reference. This experiment does not '
        'test reintroduction of revoked data, arbitrary future training, device '
        'failures or long request streams. No general no-relearning claim follows.')
    (ROOT/'manuscript/revision_continuation.tex').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    p=ROOT/'manuscript/Fed-EMNU_MDPI.tex';t=p.read_text(encoding='utf-8')
    if '\\input{revision_continuation.tex}' not in t:
        t=t.replace('\\input{revision_clients20.tex}','\\input{revision_clients20.tex}\n\\input{revision_continuation.tex}')
    t=t.replace('post-event federated training remains under investigation.',
        'five-round retained-only continuation is evaluated in Section~\\ref{sec:continuation},\nwithout establishing behavior under arbitrary future training or revoked-data reuse.')
    p.write_text(t,encoding='utf-8')
    print(output)
    print('FA increases',len(increases),'of six Fed-EMNU continuations; text tables generated, no PDF')


if __name__=='__main__':main()
