"""Combine matching core repetitions; descriptive t intervals, never pilots."""
import json
import numpy as np
from summarize_revision import ROOT, read_runs, digest


def interval(values):
    x = np.asarray(values,dtype=float)
    if len(x) != 5 or not np.isfinite(x).all():
        raise ValueError('Five finite independent seed measurements required')
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    half = 2.7764451051977987 * sd / np.sqrt(5)
    return {'n':5,'mean':mean,'sd':sd,'ci95':[mean-half,mean+half]}


def main():
    paths = [ROOT/'experiments'/name/'suite.json' for name in ('core-suite','core-extra-seeds')]
    suites = [json.loads(p.read_text()) for p in paths]
    if any(s['status'] != 'complete' or not s['spec']['controls'] for s in suites):
        raise ValueError('Both full core suites with controls must be complete')
    for field in ('settings','source_hashes','protocol_version'):
        if suites[0]['spec'][field] != suites[1]['spec'][field]:
            raise ValueError('Mismatched core protocol: '+field)
    rows = []
    for suite in suites:
        current, _, missing = read_runs(suite)
        if missing:
            raise ValueError('Unfinished run')
        rows += current
    if len(rows) != 100 or len({(r['setting'],r['seed'],r['method']) for r in rows}) != 100:
        raise ValueError('Expected 100 unique core stages')
    groups, pairs = {}, {}
    metrics = ('fa','ra','tv','target_macro_auc','seconds')
    for setting in ('iid','noniid'):
        for method in sorted({r['method'] for r in rows}):
            selected = [r for r in rows if r['setting']==setting and r['method']==method]
            if sorted(r['seed'] for r in selected) != [42,43,44,45,46]:
                raise ValueError('Wrong seed membership')
            groups[setting+'/'+method] = {m:interval([r[m] for r in selected]) for m in metrics if all(m in r for r in selected)}
        for first,second in [('selection_confidence','selection_random'),('noise','unsir'),('noise','retrain')]:
            indexed = {(r['seed'],r['method']):r for r in rows if r['setting']==setting}
            pairs[setting+'/'+first+' minus '+second] = {
                m:interval([indexed[s,first][m]-indexed[s,second][m] for s in range(42,47)])
                for m in metrics}
    output=ROOT/'experiments/five-seed-summary.json'
    output.write_text(json.dumps({'suite_hashes':[digest(p) for p in paths],
        'interval_assumptions':'Two-sided 95% Student-t, df=4, independent approximately normal seed means/differences; unadjusted descriptive intervals; no privacy guarantee',
        'summary':groups,'paired_differences':pairs},indent=2),encoding='utf-8')
    print(output)


if __name__=='__main__':
    main()
