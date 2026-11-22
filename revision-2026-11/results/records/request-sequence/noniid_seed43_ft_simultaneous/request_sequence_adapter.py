"""Cumulative revocation wrapper; canonical runner/evaluator remain unchanged."""
import argparse
import sys
from pathlib import Path
import revision_runner as runner


def validate(cumulative, current, prior):
    if len(set(cumulative)) != len(cumulative) or not set(current) <= set(cumulative):
        raise ValueError('Current request must be within distinct cumulative targets')
    if set(prior) & set(current) or set(prior) | set(current) != set(cumulative):
        raise ValueError('Prior and current requests must partition cumulative targets')


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--current_targets',nargs='+',type=int,required=True)
    parser.add_argument('--prior_targets',nargs='*',type=int,default=[])
    own, remaining=parser.parse_known_args()
    sys.argv=[sys.argv[0]]+remaining
    original_parse=runner.parse_args
    original_unlearn=runner.unlearn
    def parse():
        args=original_parse()
        validate(args.unlearn_class,own.current_targets,own.prior_targets)
        args.current_targets=own.current_targets
        args.prior_targets=own.prior_targets
        args.sequence_adapter_sha256=runner.file_hash(__file__)
        args.request_sequence_protocol_sha256=runner.file_hash(Path(__file__).resolve().parent.parent/'experiments/REQUEST_SEQUENCE_PROTOCOL.md')
        return args
    def update(model,train,partitions,retain,args,directory):
        cumulative=args.unlearn_class
        # retain was computed from cumulative revoked labels. Participation,
        # synthetic targets and GA access concern this event's new targets only.
        args.unlearn_class=own.current_targets
        try:
            result=original_unlearn(model,train,partitions,retain,args,directory)
        finally:
            args.unlearn_class=cumulative
        labels=train.tensors[1].numpy()
        if any(int(labels[i]) in cumulative for part in retain for i in part):
            raise ValueError('Cumulative quarantine violated')
        runner.json_save(directory/'sequence_protocol.json',{
            'cumulative_targets':cumulative,'current_targets':own.current_targets,
            'prior_targets':own.prior_targets,'parent_sha256':runner.file_hash(args.checkpoint),
            'reference_sha256':runner.file_hash(args.reference),
            'repair_excludes_all_cumulative_targets':True,
            'participation':'clients holding current request targets',
            'timing':'per-event; no claim of equal total simultaneous/sequential compute'})
        for source in (Path(__file__),Path(__file__).resolve().parent.parent/'experiments/REQUEST_SEQUENCE_PROTOCOL.md'):
            (directory/source.name).write_bytes(source.read_bytes())
        return result
    runner.parse_args=parse
    runner.unlearn=update
    runner.main()


if __name__=='__main__':
    main()
