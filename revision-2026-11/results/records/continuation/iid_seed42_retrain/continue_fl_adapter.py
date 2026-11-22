"""Retained-only continuation of saved states, explicitly marked semantics."""
import random
import torch
from pathlib import Path
import revision_runner as runner

original_train=runner.federated_train
def continuation(model,train,retain,old_schedule,args,directory,metadata):
    parent=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(parent['model_state'])
    rounds=1 if args.pilot else 5
    rng=random.Random(args.seed+70000)
    schedule=[rng.sample(range(args.num_clients),max(1,int(args.select_ratio*args.num_clients))) for _ in range(rounds)]
    labels=train.tensors[1]
    if any(int(labels[i]) in args.unlearn_class for part in retain for i in part):
        raise ValueError('Revoked example in continuation')
    runner.json_save(directory/'continuation_schedule.json',schedule)
    runner.json_save(directory/'continuation_protocol.json',{
        'semantics':'continued parent weights; not fresh retraining',
        'parent_sha256':runner.file_hash(args.checkpoint),'rounds':rounds,
        'quarantine':args.unlearn_class,'local_seed_offset':50000,
        'reference':'continued retained-data reference with same added budget',
        'adapter_sha256':runner.file_hash(__file__),
        'protocol_sha256':runner.file_hash(Path(__file__).resolve().parent.parent/'experiments/CONTINUATION_PROTOCOL.md')})
    (directory/'continue_fl_adapter.py').write_bytes(Path(__file__).read_bytes())
    old_seed=args.seed;args.seed+=50000
    try:return original_train(model,train,retain,schedule,args,directory,metadata)
    finally:args.seed=old_seed
def update(model,train,partitions,retain,args,directory):
    steps=continuation(model,train,retain,None,args,directory,{})
    return {'network_latency_measured':False,'note':'FL continuation tensor transport not simulated'},steps
runner.federated_train=continuation
runner.unlearn=update
if __name__=='__main__':runner.main()
