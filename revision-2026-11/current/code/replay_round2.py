"""Portable entry point: use a regenerated legacy project instead of workstation paths."""
import argparse
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-root',type=Path,required=True)
    p.add_argument('--analyze',action='store_true')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    legacy=args.legacy_root.resolve()
    if args.analyze:
        import analyze_round2 as job
        job.SOURCE=legacy
        job.main()
    else:
        import run_round2 as job
        job.SOURCE=legacy
        sys.argv=[str(Path(job.__file__))]+(['--check-only'] if args.check_only else [])
        job.main()

if __name__=='__main__': main()
