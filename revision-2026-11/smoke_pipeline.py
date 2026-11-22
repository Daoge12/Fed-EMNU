"""Small CPU/GPU pipeline check, permanently pilot=true; never scientific evidence."""
import argparse,subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    out=a.output_root.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty output-root')
    out.mkdir(parents=True,exist_ok=True);runner=ROOT/'legacy/code/revision_runner.py'
    common=['--pilot','--device',a.device,'--threads','2','--iid','--num_clients','2','--batch_size','4','--global_epochs','1','--local_epochs','1','--limit_train','40','--limit_test','40','--data_root',str(a.data_root.resolve()),'--noise_epochs','1','--noise_steps','1','--noise_copies','1','--repair_epochs','1']
    checkpoint=None;reference=None
    for mode in ('pretrain','retrain','noise'):
        cmd=[sys.executable,str(runner),'--mode',mode,'--output_dir',str(out/mode),*common]
        if checkpoint:cmd+=['--checkpoint',str(checkpoint)]
        if reference:cmd+=['--reference',str(reference)]
        with (out/(mode+'.log')).open('w',encoding='utf8') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        paths=list((out/mode).glob('*/metrics.json'));assert len(paths)==1;m=json.loads(paths[0].read_text());assert m['pilot']
        if mode=='pretrain':checkpoint=paths[0].parent/'final_model.pt'
        elif mode=='retrain':reference=paths[0].parent/'final_model.pt'
    print('PILOT_PIPELINE_OK: pretrain, fresh retrain, approximate update and evaluation; excluded from all reported results')
if __name__=='__main__':main()
