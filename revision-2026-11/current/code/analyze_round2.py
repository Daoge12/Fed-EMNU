"""Audit saved second-round outcomes and generate traceable tables."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(r'C:\Users\le\Documents\Codex\2026-11-09\ni\outputs\Fed-EMNU-revision')
OUT=ROOT/'experiments/round2'
NAMES={'fed_high':'Fed high','unsir_high':'UNSIR high','fed_low':'Fed low',
       'unsir_low':'UNSIR low','gaussian':'Gaussian','equal_event':'Equal weights'}

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def stats(x):
    x=np.asarray(x,dtype=float)
    return dict(n=len(x),mean=float(x.mean()),sd=float(x.std(ddof=1)),
                median=float(np.median(x)),q1=float(np.quantile(x,.25)),q3=float(np.quantile(x,.75)),
                min=float(x.min()),max=float(x.max()))
def fmt(x,scale=1,digits=2):
    return f'${x["mean"]*scale:.{digits}f} \\pm {x["sd"]*scale:.{digits}f}$'
def hash_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    suite=read(OUT/'suite.json')
    assert suite['status']=='complete'
    for name,digest in suite['source_hashes'].items():
        assert hash_file(ROOT/'code'/name)==digest
    runs=suite['runs']; assert len(runs)==42
    records={};rows=[];audit=[]
    for key,rec in runs.items():
        assert rec['status']=='complete'
        d=Path(rec['run_dir']); m=read(d/'metrics.json'); c=read(d/'config.json')
        assert read(d/'status.json')['status']=='complete' and not m['pilot']
        assert c['retain_ratio']==1 and c['request_scope']=='global_class'
        assert hash_file(c['checkpoint'])==c['checkpoint_sha256']
        assert hash_file(c['reference'])==m['reference_sha256']
        assert c['runner_sha256']==suite['source_hashes']['revision_runner.py']
        assert m['after']['test_count']==10000
        pred=np.load(d/'test_predictions.npz')
        assert np.isfinite(pred['probabilities']).all()
        y=pred['labels'];correct=pred['probabilities'].argmax(1)==y
        assert np.isclose(correct[y==0].mean(),m['after']['forget_acc'])
        assert np.isclose(correct[y!=0].mean(),m['after']['retain_acc'])
        v=dict(setting=rec['setting'],seed=rec['seed'],variant=rec['variant'],repeat=rec['repeat'],
               fa=m['after']['forget_acc'],ra=m['after']['retain_acc'],
               auc=m['membership']['groups']['target']['macro_auc'],
               tv=m['retrain_distance']['total_variation'],seconds=m['method_seconds'],
               steps=m['optimizer_steps'],noise_steps=c['noise_epochs']*c['noise_steps'],
               run_dir=str(d),config=c,metrics=m)
        records[key]=v
        rows.append({k:v[k] for k in ('setting','seed','variant','repeat','fa','ra','auc','tv','seconds','steps','noise_steps','run_dir')})
    for setting in ('iid','noniid'):
        for seed in (42,43,44):
            def r(name): return records[f'{setting}_seed{seed}_{name}_0']
            base=r('fed_high')
            for name in NAMES:
                other=r(name)
                for f in ('checkpoint_sha256','reference','unlearn_seed','retain_ratio','selection','impair_lr','repair_lr'):
                    assert base['config'][f]==other['config'][f],(setting,seed,name,f)
                assert read(Path(base['run_dir'])/'selection.json')==read(Path(other['run_dir'])/'selection.json')
                assert read(Path(base['run_dir'])/'attack_splits.json')==read(Path(other['run_dir'])/'attack_splits.json')
            for a,b in [('fed_high','unsir_high'),('fed_low','unsir_low')]:
                assert r(a)['steps']==r(b)['steps'] and r(a)['noise_steps']==r(b)['noise_steps']
            for name in ('gaussian','equal_event'):
                assert r(name)['steps']==base['steps']
            if setting=='iid':
                a=np.load(Path(base['run_dir'])/'test_predictions.npz')['probabilities']
                b=np.load(Path(r('equal_event')['run_dir'])/'test_predictions.npz')['probabilities']
                audit.append(dict(check='IID event weighting equality',seed=seed,max_abs=float(np.abs(a-b).max())))
                assert np.array_equal(a,b)
    summary={}
    for setting in ('iid','noniid'):
        for name in NAMES:
            group=[v for v in records.values() if v['setting']==setting and v['variant']==name]
            assert len(group)==3
            summary[f'{setting}/{name}']={f:stats([v[f] for v in group]) for f in ('fa','ra','auc','tv','seconds','steps','noise_steps')}
    paired={}
    for setting in ('iid','noniid'):
        for a,b in [('fed_high','unsir_high'),('fed_low','unsir_low'),('fed_high','gaussian'),('fed_high','equal_event')]:
            paired[f'{setting}/{a}-{b}']={f:stats([records[f'{setting}_seed{seed}_{a}_0'][f]-records[f'{setting}_seed{seed}_{b}_0'][f]
                                                for seed in (42,43,44)]) for f in ('fa','ra','auc','seconds')}
    old={}
    for name in ('core-suite','core-extra-seeds'):
        old.update(read(SOURCE/f'experiments/{name}/suite.json')['runs'])
    timing={}; legacy=[]
    for setting in ('iid','noniid'):
        for mode in ('retrain','noise','unsir'):
            values=[]
            for seed in (42,43,44,45,46):
                d=Path(old[f'{setting}_seed{seed}_{mode}']['run_dir'])
                m=read(d/'metrics.json'); values.append(m['method_seconds'])
                legacy.append(dict(setting=setting,seed=seed,mode=mode,seconds=m['method_seconds'],run_dir=str(d)))
            timing[f'{setting}/{mode}']=stats(values)
    repeat={}
    for seed in (42,45):
        group=[v for v in records.values() if v['variant']=='timing_repeat' and v['seed']==seed]
        assert len(group)==3
        original=Path(old[f'noniid_seed{seed}_noise']['run_dir'])
        original_p=np.load(original/'test_predictions.npz')['probabilities']
        comparisons=[]
        for v in group:
            pp=np.load(Path(v['run_dir'])/'test_predictions.npz')['probabilities']
            comparisons.append(dict(repeat=v['repeat'],max_abs=float(np.abs(pp-original_p).max()),
                prediction_disagreement=float((pp.argmax(1)!=original_p.argmax(1)).mean())))
        repeat[str(seed)]={'seconds':stats([v['seconds'] for v in group]),'prediction_comparison':comparisons,
                          'legacy_seconds':read(original/'metrics.json')['method_seconds']}
    result=dict(summary=summary,paired=paired,legacy_timing=timing,technical_repeats=repeat,
                audits=audit,new_scientific_control_events=36,technical_repeat_events=6,
                suite_sha256=hash_file(OUT/'suite.json'))
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for name,data in [('seed-level.csv',rows),('legacy-timing.csv',legacy)]:
        with (OUT/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
    tex=r'''\subsection{Controlled Noise Objectives and Event Aggregation}
\label{sec:round2-controls}
The second-round controls use the original CIFAR-10/ResNet18 checkpoints and
references for seeds 42--44 in both partitions. All controls retain the full
local retained set and use the same random ordering, local learning rates and
attack splits. This explicitly removes the mode-dependent automatic ordering
used in the primary recipes. The high budget has 80 server noise steps and
three repair epochs; the low budget has 40 steps and one repair epoch. One
impair epoch is used throughout. At each budget, Fed and UNSIR differ only in
the norm penalty defined in Section~\ref{sec:noise}. Realized local optimizer
steps are identical within each paired budget. These are matched update
budgets, not enforced equal wall-clock times or the centralized UNSIR protocol.
The Gaussian control omits noise optimization; equal weights changes only
request-event aggregation. Both otherwise follow Fed high. Gaussian saves
server steps and is not equal compute. No result was used to tune a budget.
Values are mean $\pm$ sample SD across three paired scientific seeds.
'''
    for setting,title in [('iid','IID'),('noniid','Non-IID')]:
        tex+=r'\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{4pt}'+'\n'
        tex+=r'\caption{Second-round '+title+r' controls ($n=3$). Steps count local model updates; server steps are specified in the text.}'+'\n'
        tex+=r'\label{tab:round2-'+setting+r'}\begin{tabular}{lrrrrrr}\toprule Variant & FA (\%) & RA (\%) & Target AUC & TV & Steps & Time (s)\\\midrule'+'\n'
        for name,label in NAMES.items():
            g=summary[f'{setting}/{name}']
            tex+=label+' & '+' & '.join([fmt(g['fa'],100),fmt(g['ra'],100),fmt(g['auc'],digits=3),fmt(g['tv'],digits=3),
                                          fmt(g['steps'],digits=0),fmt(g['seconds'])])+r'\\'+'\n'
        tex+=r'\bottomrule\end{tabular}\end{table}'+'\n'
    # Interpretation is added after reviewing this numerical output.
    tex+='\\input{revision_round2_interpretation.tex}\n'
    tex+=r'''\subsection{Robust Timing Summaries and Outlier Investigation}
\label{sec:round2-timing}
Table~\ref{tab:robust-time} supplements the original mean/SD tables with medians
and quartiles from the same five primary seeds, retaining every observation.
Quartiles use linear interpolation. These seed-level times include changes in
partition/schedule and execution conditions; they are not identical-workload
technical repetitions. Historical hardware telemetry is unavailable, so a
specific cause of the old outlier cannot be established from saved times.
\begin{table}[H]\centering\small
\caption{Original primary compute/serialization time in seconds, five seeds;
median [first, third quartile]. Evaluation and network transport excluded.}
\label{tab:robust-time}\begin{tabular}{llrr}\toprule
Setting & Method & Median [Q1, Q3] & Range\\\midrule
'''
    for setting,title in [('iid','IID'),('noniid','Non-IID')]:
        for mode,label in [('retrain','Retrain'),('noise','Fed-EMNU'),('unsir','UNSIR-style')]:
            g=timing[f'{setting}/{mode}']
            tex+=f'{title} & {label} & {g["median"]:.2f} [{g["q1"]:.2f}, {g["q3"]:.2f}] & {g["min"]:.2f}--{g["max"]:.2f}'+r'\\'+'\n'
    tex+=r'\bottomrule\end{tabular}\end{table}'+'\n'
    slow_retrain=read(Path(old['noniid_seed45_retrain']['run_dir'])/'metrics.json')['method_seconds']
    slow_unsir=read(Path(old['noniid_seed45_unsir']['run_dir'])/'metrics.json')['method_seconds']
    others=[read(Path(old[f'noniid_seed{seed}_retrain']['run_dir'])/'metrics.json')['method_seconds'] for seed in (42,43,44,46)]
    tex+=f'The historical delay was not confined to Fed-EMNU: Non-IID seed 45 also\n'
    tex+=f'took {slow_retrain:.2f} s for retraining (other seeds {min(others):.2f}--{max(others):.2f} s)\n'
    tex+=f'and {slow_unsir:.2f} s for the UNSIR-style event. This cross-method pattern\n'
    tex+='documents a broader timing anomaly, without identifying its cause.\n'
    for seed in (42,45):
        g=repeat[str(seed)];q=g['seconds']
        tex+=f'For the Non-IID seed-{seed} checkpoint, three exact-recipe technical repeats\n'
        tex+=f'took a median {q["median"]:.2f} s (range {q["min"]:.2f}--{q["max"]:.2f} s),\n'
        tex+=f'compared with {g["legacy_seconds"]:.2f} s in the original session.\n'
        delta=max(v['max_abs'] for v in g['prediction_comparison'])
        dis=max(v['prediction_disagreement'] for v in g['prediction_comparison'])
        tex+=f'The maximum posterior difference from that original result was {delta:.3g};\n'
        tex+=f'the maximum predicted-label disagreement was {100*dis:.3f}\\%.\n'
    tex+=r'''These repeats use the current driver/session and are reported separately,
without replacing or pooling the historical measurements. They test whether
the saved workload consistently reproduces the large delay, but cannot identify
the historical cause. Three repeats per checkpoint are technical replicates,
not additional independent seeds. Compute time still excludes end-to-end
federated transport, energy and deployment overhead.
'''
    (ROOT/'manuscript/revision_round2_controls.tex').write_text(tex,encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
