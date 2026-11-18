"""Serial, restartable execution of the frozen revision protocol on one GPU."""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def save(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--settings", nargs="+", choices=["iid", "noniid"], default=["iid", "noniid"])
    parser.add_argument("--controls", action="store_true", help="Also run matched selection and stage controls")
    parser.add_argument("--output", default="experiments/core-suite")
    parser.add_argument("--model", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--data", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--num-clients", type=int, default=10)
    args = parser.parse_args()
    if args.dirichlet_alpha <= 0 or args.num_clients < 1:
        parser.error("Dirichlet alpha and client count must be positive")
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: hashlib.sha256((ROOT / "code" / name).read_bytes()).hexdigest()
                     for name in ("revision_runner.py", "revision_evaluation.py")}
    # Preserve the original core manifest schema for unchanged defaults.
    extension = {name: getattr(args, name) for name, default in
                 (("model", "resnet18"), ("data", "cifar10"),
                  ("dirichlet_alpha", 0.5), ("num_clients", 10))
                 if getattr(args, name) != default}
    spec = {name: value for name, value in vars(args).items()
            if name not in ("model", "data", "dirichlet_alpha", "num_clients")}
    if extension:
        spec["extension"] = extension
    spec |= {"source_hashes": source_hashes, "protocol_version": 2}
    path = output / "suite.json"
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["spec"] != spec:
            raise ValueError("Existing suite differs; choose a new output directory")
    else:
        state = {"spec": spec, "status": "running", "runs": {}, "started": datetime.now(timezone.utc).isoformat()}
        save(path, state)

    def run(key, mode, common, extra=()):
        if key in state["runs"]:
            old = state["runs"][key]
            if old["status"] == "complete" and (Path(old["run_dir"]) / "metrics.json").exists():
                print(f"REUSE {key}: {old['run_dir']}", flush=True)
                return Path(old["run_dir"]) / "final_model.pt"
        stage_output = output / key
        command = [sys.executable, "-u", str(ROOT / "code/revision_runner.py"), "--mode", mode,
                   "--data_root", str(ROOT / "code/data"), "--output_dir", str(stage_output), *common, *extra]
        record = {"command": command, "status": "running", "started": datetime.now(timezone.utc).isoformat()}
        state["runs"][key] = record
        state["status"] = "running"
        save(path, state)
        print(f"START {key}", flush=True)
        with (output / "suite.log").open("a", encoding="utf-8", buffering=1) as log:
            process = subprocess.Popen(command, cwd=ROOT / "code", stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8")
            for line in process.stdout:
                log.write(line)
                # All full metrics remain in per-run files; keep live output compact.
                if line.startswith(("RUN_DIR=", "round ", "noise:", "unsir:", "ft:", "ga:", "Traceback")):
                    print(line.rstrip(), flush=True)
            code = process.wait()
        record["finished"] = datetime.now(timezone.utc).isoformat()
        dirs = sorted(stage_output.glob("*/status.json"))
        if code or not dirs or json.loads(dirs[-1].read_text())["status"] != "complete":
            record.update(status="failed", exit_code=code)
            state["status"] = "failed"
            save(path, state)
            raise RuntimeError(f"Stage {key} failed; inspect suite.log and stage errors.log")
        run_dir = dirs[-1].parent
        record.update(status="complete", run_dir=str(run_dir))
        save(path, state)
        print(f"COMPLETE {key}", flush=True)
        return run_dir / "final_model.pt"

    for setting in args.settings:
        for seed in args.seeds:
            prefix = f"{setting}_seed{seed}"
            common = ["--seed", str(seed), "--split_seed", str(seed), "--unlearn_seed", str(seed),
                      "--retrain_seed", str(1000 + seed)]
            for option, value in extension.items():
                common += ["--" + option, str(value)]
            if setting == "iid":
                common += ["--iid"]
            if args.pilot:
                common += ["--pilot", "--global_epochs", "1", "--local_epochs", "1",
                           "--limit_train", "1000", "--limit_test", "500"]
            checkpoint = run(prefix + "_pretrain", "pretrain", common)
            shared = ["--checkpoint", str(checkpoint)]
            reference = run(prefix + "_retrain", "retrain", common, shared)
            shared += ["--reference", str(reference)]
            for mode in ("ft", "ga", "noise", "unsir"):
                extra = ["--noise_epochs", "5", "--repair_epochs", "1"] if mode == "unsir" else []
                run(prefix + "_" + mode, mode, common, shared + extra)
            if args.controls:
                for selection in ("random", "confidence"):
                    run(prefix + "_selection_" + selection, "noise", common,
                        shared + ["--retain_ratio", "0.5", "--selection", selection])
                for ablation in ("impair_only", "repair_only"):
                    run(prefix + "_" + ablation, "noise", common, shared + ["--ablation", ablation])
    state.update(status="complete", finished=datetime.now(timezone.utc).isoformat())
    save(path, state)
    print(f"SUITE_COMPLETE={path}", flush=True)


if __name__ == "__main__":
    main()
