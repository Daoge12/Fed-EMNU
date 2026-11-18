"""Rebuild wall-clock tables from evidence, using portable suite-relative paths."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/walltime'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def stats(values):
    x=np.asarray(values,dtype=float)
    return {'n':len(x),'mean':float(x.mean()),'sd':float(x.std(ddof=1)),
            'min':float(x.min()),'max':float(x.max())}
def fmt(g,scale=1,digits=2):return f'${g["mean"]*scale:.{digits}f} \\pm {g["sd"]*scale:.{digits}f}$'
def main():
    state=read(OUT/'suite.json');assert state['status']=='complete'
    assert len(state['runs'])==48
    rows=[];saved={};audit=[]
    for key,rec in state['runs'].items():
        assert rec['status']=='complete'
        paths=list((OUT/key).glob('*/metrics.json'));assert len(paths)==1
        d=paths[0].parent;m=read(paths[0]);c=read(d/'config.json');t=read(d/'walltime_trace.json')
        assert not m['pilot'] and read(d/'status.json')['status']=='complete'
        assert t['budget_seconds']==rec['budget']
        pp=np.load(d/'test_predictions.npz');y=pp['labels'];p=pp['probabilities']
        assert len(y)==10000 and np.isfinite(p).all()
        hit=p.argmax(1)==y
        for f,v in [('forget_acc',hit[y==0].mean()),('retain_acc',hit[y!=0].mean())]:assert np.isclose(v,m['after'][f])
        assert len(t['clients'])==20
        assert sum(v['steps'] for v in t['clients'])==m['optimizer_steps']
        assert sorted(set(v['client'] for v in t['clients']))==m['communication']['participating_clients']
        for source in ('walltime_adapter.py','revision_runner.py','revision_evaluation.py'):
            assert hashlib.sha256((d/source).read_bytes()).hexdigest()==state['hashes'][source]
        row={k:rec[k] for k in ('setting','seed','budget','mode')}
        row.update(fa=m['after']['forget_acc'],ra=m['after']['retain_acc'],
                   auc=m['membership']['groups']['target']['macro_auc'],tv=m['retrain_distance']['total_variation'],
                   seconds=m['method_seconds'],steps=m['optimizer_steps'],
                   overrun_seconds=max(0.,m['method_seconds']-rec['budget']),
                   zero_step_stages=sum(v['steps']==0 for v in t['clients']),relative_run_dir=d.relative_to(ROOT).as_posix())
        rows.append(row);saved[key]=(d,m,c,t)
    paired={};groups={}
    for setting in ('iid','noniid'):
        for budget in (10,20,30,40):
            for mode in ('noise','unsir'):
                rr=[v for v in rows if v['setting']==setting and v['budget']==budget and v['mode']==mode]
                assert len(rr)==3
                groups[f'{setting}/{budget}/{mode}']={f:stats([v[f] for v in rr]) for f in ('fa','ra','auc','tv','seconds','steps')}
            delta={f:[] for f in ('fa','ra','auc','seconds')}
            for seed in (42,43,44):
                da,ma,ca,ta=saved[f'{setting}_seed{seed}_{budget}s_noise']
                db,mb,cb,tb=saved[f'{setting}_seed{seed}_{budget}s_unsir']
                for f in ('checkpoint_sha256','selection','retain_ratio','unlearn_seed'):
                    assert ca[f]==cb[f]
                assert read(da/'selection.json')==read(db/'selection.json')
                assert read(da/'attack_splits.json')==read(db/'attack_splits.json')
                va=next(v for v in rows if v['relative_run_dir']==da.relative_to(ROOT).as_posix())
                vb=next(v for v in rows if v['relative_run_dir']==db.relative_to(ROOT).as_posix())
                for f in delta:delta[f].append(va[f]-vb[f])
            paired[f'{setting}/{budget}']={f:stats(v) for f,v in delta.items()}
    violations=[r for r in rows if r['overrun_seconds']>0]
    result={'events':48,'independent_seeds_per_cell':3,'summary':groups,'paired_fed_minus_unsir':paired,
            'budget_violations':violations,'zero_step_stage_total':sum(r['zero_step_stages'] for r in rows),
            'audit':'saved predictions, source hashes, stage accounting, paired indices and all-client participation passed'}
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (OUT/'seed-level.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    tex=r'''\subsection{Common Wall-Clock Caps}
\label{sec:walltime}
To supplement matched update counts, we compare time-capped adaptations of the
two original recipes at 10, 20, 30 and 40 seconds on the same GPU, checkpoints,
retained-data ordering and references (three seeds, both partitions). The
noise and maximum repair budgets remain 80 steps/three epochs for Fed-EMNU and
40 steps/one epoch for UNSIR-style. Method order is counterbalanced by seed
and budget. No test outcome chooses a configuration or stopping point.

After noise generation, a predefined reserve of the larger of 0.5 seconds and
15\% of the total cap is withheld for non-training overhead. The remaining
local-training time is divided equally across all affected clients. Each
client divides its allocation between impair and repair in ratio
$1:\text{repair epochs}$. Updates stop at synchronized minibatch boundaries;
the largest previous minibatch duration is reserved before another batch.
Original epoch limits also apply: recipes finishing early receive no padding
or extra training. Total observed time includes noise, local updates, model
copies, selection, aggregation and final serialization, excluding evaluation
and network transport. Per-batch synchronization introduces monitoring overhead.
These controls compare common maximum resources under a specified scheduler;
they do not establish an optimal allocation or identical consumed time.
'''
    for setting,title in [('iid','IID'),('noniid','Non-IID')]:
        tex+=r'\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{4pt}'+'\n'
        tex+=r'\caption{Common time caps, '+title+r', three seeds. Times are observed compute/serialization seconds, not padded cap values.}\label{tab:walltime-'+setting+r'}'+'\n'
        tex+=r'\begin{tabular}{rlrrrrr}\toprule Cap (s) & Recipe & FA (\%) & RA (\%) & Target AUC & TV & Time (s)\\\midrule'+'\n'
        for budget in (10,20,30,40):
            for mode,label in [('noise','Fed-EMNU'),('unsir','UNSIR-style')]:
                g=groups[f'{setting}/{budget}/{mode}']
                tex+=f'{budget} & {label} & '+' & '.join([fmt(g['fa'],100),fmt(g['ra'],100),fmt(g['auc'],digits=3),fmt(g['tv'],digits=3),fmt(g['seconds'])])+r'\\'+'\n'
        tex+=r'\bottomrule\end{tabular}\end{table}'+'\n'
    tex+=f'All 48 events were checked against their requested caps: {len(violations)}\n'
    tex+='events exceeded the cap. '+f'The stage records contain {result["zero_step_stage_total"]} zero-update stages.\n'
    if violations:tex+='Exact overruns are retained in the supplementary seed-level CSV; over-budget outputs are not described as feasible under their caps.\n'
    tex+=r'''FA/RA are recomputed from saved predictions; paired comparisons use the
same checkpoints, selections and attack splits. Exact update counts and client
stage times are provided in the supplementary CSV and traces. The three seeds
per cell limit inference. Truncation changes stage completion and thus the
original recipes; these results cannot be substituted for their unrestricted
defaults or used as proof of general superiority at every time allocation.
\input{revision_walltime_interpretation.tex}
'''
    (ROOT/'manuscript/revision_walltime.tex').write_text(tex,encoding='utf-8')
    print('PASS',len(rows),'events; cap violations',len(violations))
    for k,v in paired.items():print(k,'RA pp',v['ra']['mean']*100,'FA pp',v['fa']['mean']*100,'AUC',v['auc']['mean'])
if __name__=='__main__':main()
