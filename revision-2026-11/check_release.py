"""CPU-only coverage/hash/numerical-record checks; does not inspect raw predictions."""
from pathlib import Path
import json,hashlib,csv,math,statistics
from reproduce import plan,ORDER
ROOT=Path(__file__).resolve().parent
def close(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-10),(a,b)
def main():
    manifest=json.loads((ROOT/'PUBLIC_MANIFEST.json').read_text())
    for name,digest in manifest.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    rows,_=plan(ORDER);assert len(rows)==412
    lookup={};counts={}
    for r in rows:
        d=ROOT/r['record'];m=json.loads((d/'metrics.json').read_text());c=json.loads((d/'config.json').read_text())
        assert not m['pilot'] and not c['pilot'];counts[r['suite']]=counts.get(r['suite'],0)+1
        for n,h in r['sources'].items():assert hashlib.sha256((d/n).read_bytes()).hexdigest()==h
        assert r['script'] in r['sources']
        assert c['runner_sha256']==r['sources']['revision_runner.py']
        assert c['evaluation_sha256']==r['sources']['revision_evaluation.py']
        lookup[r['suite'],r['key']]=(r,m,c)
    for suite in ('round2','walltime'):
        with (ROOT/'results'/suite/'seed-level.csv').open(encoding='utf8') as f:
            for row in csv.DictReader(f):
                key=(f"{row['setting']}_seed{row['seed']}_{row['variant']}_{row['repeat']}" if suite=='round2' else f"{row['setting']}_seed{row['seed']}_{row['budget']}s_{row['mode']}")
                _,m,c=lookup[suite,key]
                for a,b in [(row['fa'],m['after']['forget_acc']),(row['ra'],m['after']['retain_acc']),(row['auc'],m['membership']['groups']['target']['macro_auc']),(row['tv'],m['retrain_distance']['total_variation']),(row['seconds'],m['method_seconds']),(row['steps'],m['optimizer_steps'])]:close(a,b)
                if suite=='walltime':assert m['method_seconds']<=int(row['budget'])
    summary=json.loads((ROOT/'results/legacy/five-seed-summary.json').read_text())
    # Primary CIFAR-10 means are independently recomputed from the public metrics.
    for setting in ('iid','noniid'):
        vals=[]
        for seed in range(42,47):
            suite='core-suite' if seed<45 else 'core-extra-seeds';_,m,_=lookup[suite,f'{setting}_seed{seed}_noise'];vals.append(m['after']['retain_acc'])
        close(statistics.mean(vals),summary['summary'][setting+'/noise']['ra']['mean'])
        close(statistics.stdev(vals),summary['summary'][setting+'/noise']['ra']['sd'])
        print(setting,'Fed-EMNU RA',statistics.mean(vals),'SD',statistics.stdev(vals))
    print(json.dumps({'manifest_files':len(manifest),'formal_records':len(rows),'counts':counts,'control_csv_records_checked':90,'status':'passed'},indent=2))
if __name__=='__main__':main()
