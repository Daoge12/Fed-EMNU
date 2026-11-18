"""Replay against any regenerated legacy root without editing frozen experiment sources."""
import argparse
from pathlib import Path
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--legacy-root',type=Path,required=True)
    args=p.parse_args()
    import run_walltime as job
    job.SOURCE=args.legacy_root.resolve()
    job.main()
if __name__=='__main__':main()
