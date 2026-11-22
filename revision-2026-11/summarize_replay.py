"""Summarize regenerated outputs without mixing methods, variants or technical repeats."""
import argparse,json,csv,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def stat(v):return {'n':len(v),'mean':statistics.mean(v),'sd':statistics.stdev(v) if len(v)>1 else None}
def summarize(rows):
    groups={}
    fields=('fa','ra','target_auc','tv','actual_seconds','local_steps')
    for r in rows:
        key='/'.join(str(r[k]) for k in ('suite','setting','variant','budget','repeat'))
        groups.setdefault(key,[]).append(r)
    summary={k:{f:stat([r[f] for r in rs]) for f in fields if all(r[f] is not None for r in rs)} for k,rs in groups.items()}
    paired={}
    for setting in ('iid','noniid'):
        for a,b in [('fed_high','unsir_high'),('fed_low','unsir_low'),('fed_high','gaussian'),('fed_high','equal_event')]:
            x={r['seed']:r for r in rows if r['suite']=='round2' and r['setting']==setting and r['variant']==a and r['repeat']==0}
            y={r['seed']:r for r in rows if r['suite']=='round2' and r['setting']==setting and r['variant']==b and r['repeat']==0}
            seeds=sorted(x.keys() & y.keys())
            if seeds:paired[setting+'/'+a+'-minus-'+b]={f:stat([x[s][f]-y[s][f] for s in seeds]) for f in fields if all(x[s][f] is not None and y[s][f] is not None for s in seeds)}
    return {'grouped_summary':summary,'paired_controls':paired}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=ROOT/'replay-output');p.add_argument('--published',action='store_true',help='Summarize public historical metrics instead of regenerated outputs');a=p.parse_args()
    catalog=json.loads((ROOT/'protocols/replay-catalog.json').read_text());state={} if a.published else json.loads((a.output_root/'replay-state.json').read_text());rows=[]
    for r in catalog:
        ident=r['suite']+'/'+r['key']
        if not a.published and ident not in state:continue
        d=ROOT/r['record'] if a.published else Path(state[ident]['run_dir']);m=json.loads((d/'metrics.json').read_text());c=json.loads((d/'config.json').read_text());assert not m['pilot']
        if r['suite']=='round2':variant=r['key'].split('_',2)[2].rsplit('_',1)[0];repeat=int(r['key'].rsplit('_',1)[1])
        elif r['suite']=='walltime':variant=c['mode'];repeat=0
        else:variant=r['key'].split('_',2)[2];repeat=0
        budget=next((r['flags'][i+1] for i,x in enumerate(r['flags']) if x=='--wall_budget'),'native')
        rows.append(dict(suite=r['suite'],event=r['key'],setting='iid' if c['iid'] else 'noniid',seed=c['seed'],variant=variant,budget=budget,repeat=repeat,fa=m['after']['forget_acc'],ra=m['after']['retain_acc'],target_auc=m.get('membership',{}).get('groups',{}).get('target',{}).get('macro_auc'),tv=m.get('retrain_distance',{}).get('total_variation'),actual_seconds=m['method_seconds'],local_steps=m.get('optimizer_steps')))
    if not rows:raise ValueError('No completed formal outputs')
    a.output_root.mkdir(parents=True,exist_ok=True)
    with (a.output_root/'seed-level-regenerated.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.output_root/'summary-regenerated.json').write_text(json.dumps(summarize(rows),indent=2),encoding='utf8')
    print('SUMMARIZED',len(rows),'formal records; technical repeats kept separate')
if __name__=='__main__':main()
