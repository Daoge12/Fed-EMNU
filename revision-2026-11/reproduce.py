"""Replay recorded formal commands in dependency order; defaults to a dry run."""
import argparse,json,re,sys,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ORDER=['core-suite','core-extra-seeds','extension-resnet34','extension-alpha01',
       'extension-cifar100','extension-clients20','par-comparison','request-sequence',
       'continuation','round2','walltime']
def norm(s):return re.sub('/+','/',s.replace('\\','/'))
def records():
    rows=json.loads((ROOT/'protocols/replay-catalog.json').read_text(encoding='utf8'))
    return sorted(enumerate(rows),key=lambda x:(ORDER.index(x[1]['suite']),x[0]))
def plan(selected):
    allrows=[r for _,r in records()];byrun={norm(r['old_run']):r for r in allrows};required=set()
    def visit(r):
        ident=(r['suite'],r['key'])
        if ident in required:return
        required.add(ident)
        for i,arg in enumerate(r['flags']):
            if arg in ('--checkpoint','--reference'):
                old=norm(r['flags'][i+1]);parent=old.rsplit('/',1)[0]
                if parent not in byrun:raise ValueError('Unmapped checkpoint '+old)
                visit(byrun[parent])
    for r in allrows:
        if r['suite'] in selected:visit(r)
    return [r for r in allrows if (r['suite'],r['key']) in required],byrun
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',nargs='+',choices=ORDER+['all'],default=['all'])
    p.add_argument('--output-root',type=Path,default=ROOT/'replay-output')
    p.add_argument('--data-root',type=Path,default=ROOT/'data')
    p.add_argument('--execute',action='store_true',help='Run full scientific recipes; requires datasets, CUDA and substantial time')
    args=p.parse_args();selected=ORDER if 'all' in args.suite else args.suite
    rows,byrun=plan(selected);out=args.output_root.resolve();data=args.data_root.resolve()
    statepath=out/'replay-state.json';state=json.loads(statepath.read_text()) if statepath.exists() else {}
    for r in rows:
        ident=r['suite']+'/'+r['key'];stage=out/r['suite']/r['key'];sources=ROOT/r['record']
        script=stage/'source'/r['script'];flags=list(r['flags'])
        for i,arg in enumerate(flags):
            if arg in ('--checkpoint','--reference'):
                prior=byrun[norm(flags[i+1]).rsplit('/',1)[0]];pid=prior['suite']+'/'+prior['key']
                flags[i+1]=str(Path(state[pid]['run_dir'])/'final_model.pt') if pid in state else '<generated:'+pid+'/final_model.pt>'
            elif arg=='--output_dir':flags[i+1]=str(stage/'runs')
            elif arg=='--data_root':flags[i+1]=str(data)
        cmd=[sys.executable,'-u',str(script),*flags]
        if not args.execute:print(json.dumps({'event':ident,'command':cmd},ensure_ascii=False));continue
        if state.get(ident,{}).get('status')=='complete':
            if not (Path(state[ident]['run_dir'])/'final_model.pt').is_file():raise FileNotFoundError('Resume checkpoint missing: '+ident)
            continue
        if any(s.startswith('<generated:') for s in flags):raise RuntimeError('Dependency missing for '+ident)
        if not data.exists():raise FileNotFoundError('Prepare official CIFAR data first: '+str(data))
        script.parent.mkdir(parents=True,exist_ok=True)
        for name in r['sources']:shutil.copy2(sources/name,script.parent/name)
        if not script.exists():raise FileNotFoundError(script)
        print('START',ident,flush=True)
        with (stage/'execution.log').open('w',encoding='utf8') as log:
            subprocess.run(cmd,cwd=script.parent,stdout=log,stderr=subprocess.STDOUT,check=True)
        candidates=list((stage/'runs').glob('*/metrics.json'))
        if len(candidates)!=1:raise RuntimeError('Expected one fresh run; use a fresh output-root: '+ident)
        state[ident]={'status':'complete','run_dir':str(candidates[0].parent),'command':cmd}
        statepath.write_text(json.dumps(state,indent=2),encoding='utf8')
        print('COMPLETE',ident,flush=True)
    print(json.dumps({'planned_events':len(rows),'executed':args.execute}))
if __name__=='__main__':main()
