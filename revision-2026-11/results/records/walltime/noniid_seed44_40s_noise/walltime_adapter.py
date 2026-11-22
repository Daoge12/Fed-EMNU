"""Deadline-controlled adaptation of the frozen runner; original recipes remain intact."""
import sys
import time
import shutil
from pathlib import Path
import torch
import revision_runner as r

TRACE=[]
BUDGET=float(sys.argv[sys.argv.index('--wall_budget')+1])
i=sys.argv.index('--wall_budget');del sys.argv[i:i+2]
if BUDGET <= 0: raise ValueError('Budget must be positive')
ORIGINAL_UNLEARN=r.unlearn
DEADLINES={}

def timed_train(model,dataset,indices,args,epochs,lr,seed,ascent=False):
    # Each client has an independent deadline; no test outcomes choose a stop.
    cid=seed-args.unlearn_seed
    entry=DEADLINES[cid]
    stage='impair' if not entry['impair_done'] else 'repair'
    entry['impair_done']=True
    start=time.perf_counter()
    deadline=start+entry['allocation']*(1/(1+args.repair_epochs) if stage=='impair' else args.repair_epochs/(1+args.repair_epochs))
    batches=r.loader(dataset,indices,args,shuffle=True,seed=seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    steps=0;largest_step=0.
    for _ in range(epochs):
        for x,y in batches:
            r.synchronize(args)
            if time.perf_counter()+largest_step >= deadline: break
            t=time.perf_counter()
            model.train(len(x)>1);optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x.to(args.resolved_device)),y.to(args.resolved_device))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite local loss')
            loss.backward();optimizer.step();r.synchronize(args)
            largest_step=max(largest_step,time.perf_counter()-t);steps+=1
        else:continue
        break
    TRACE.append(dict(client=cid,stage=stage,steps=steps,allocated_seconds=deadline-start,
                      actual_seconds=time.perf_counter()-start))
    return steps

def timed_unlearn(model,train,partitions,retain,args,run_dir):
    # Noise, selection, aggregation and serialization are included in total time.
    start=time.perf_counter()
    labels=train.tensors[1].numpy()
    affected=[cid for cid in range(args.num_clients) if any(int(labels[j]) in args.unlearn_class for j in partitions[cid])]
    original_noise=r.optimized_noise
    def noise(*a,**kw):
        result=original_noise(*a,**kw)
        r.synchronize(args)
        # Reserve 15% (minimum 0.5 s) for model copies, selection, aggregation,
        # synchronization and final serialization; unused capacity is reported.
        remain=max(0.,BUDGET-(time.perf_counter()-start)-max(.5,.15*BUDGET))
        for cid in affected:DEADLINES[cid]={'allocation':remain/len(affected),'impair_done':False}
        return result
    r.optimized_noise=noise;r.train_epochs=timed_train
    args.wall_budget_seconds=BUDGET
    result=ORIGINAL_UNLEARN(model,train,partitions,retain,args,run_dir)
    r.json_save(run_dir/'walltime_trace.json',dict(budget_seconds=BUDGET,clients=TRACE,
               allocation_rule='remaining after noise minus max(0.5s,15% total), equally across clients; impair:repair=1:repair_epochs',
               purpose='time-capped original recipes; no extra epochs after recipe completion'))
    shutil.copy2(__file__,run_dir/'walltime_adapter.py')
    return result

if __name__=='__main__':
    r.unlearn=timed_unlearn
    r.main()
