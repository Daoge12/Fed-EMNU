"""Canonical revision runner with explicit scopes and auditable artifacts.

Run `python revision_runner.py --help`. The original modules are retained as
historical implementation context; this runner has no import-time CLI globals.
"""
import argparse
import copy
import hashlib
import json
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset
import torchvision
from torchvision import datasets, models

from revision_evaluation import (posterior, accuracy_metrics, prediction_distance,
                                 make_attack_splits, membership_diagnostics)


class Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, text):
        self.file.write(text)
        return self.stream.write(text)

    def flush(self):
        self.file.flush()
        self.stream.flush()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["pretrain", "noise", "unsir", "retrain", "ft", "ga"], required=True)
    p.add_argument("--model", choices=["resnet18", "resnet34"], default="resnet18")
    p.add_argument("--data", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--output_dir", default="../experiments/runs")
    p.add_argument("--checkpoint")
    p.add_argument("--reference", help="Saved from-scratch reference for prediction/MIA differences")
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--select_ratio", type=float, default=0.4)
    p.add_argument("--global_epochs", type=int, default=50, help="Federated rounds, also used by Retrain")
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--iid", action="store_true")
    p.add_argument("--dirichlet_alpha", type=float, default=0.5)
    p.add_argument("--aggregation", choices=["samples", "equal"], default="samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--unlearn_seed", type=int, default=42)
    p.add_argument("--retrain_seed", type=int, default=1042)
    p.add_argument("--attack_seed", type=int, default=90210)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--unlearn_class", type=int, nargs="+", default=[0])
    p.add_argument("--client_id", type=int, default=0)
    p.add_argument("--request_scope", choices=["requesting_client", "global_class"], default="global_class")
    p.add_argument("--retain_ratio", type=float, default=1.0)
    p.add_argument("--selection", choices=["auto", "random", "confidence"], default="auto")
    p.add_argument("--noise_copies", type=int, default=20)
    p.add_argument("--noise_epochs", type=int, default=10)
    p.add_argument("--noise_steps", type=int, default=8)
    p.add_argument("--noise_lr", type=float, default=0.1)
    p.add_argument("--noise_l2", type=float, default=0.1)
    p.add_argument("--impair_lr", type=float, default=0.02)
    p.add_argument("--repair_lr", type=float, default=0.01)
    p.add_argument("--repair_epochs", type=int, default=3)
    p.add_argument("--ablation", choices=["full", "impair_only", "repair_only"], default="full")
    p.add_argument("--ft_lr", type=float, default=0.001)
    p.add_argument("--ft_epochs", type=int, default=5)
    p.add_argument("--ga_lr", type=float, default=0.001)
    p.add_argument("--ga_epochs", type=int, default=5)
    p.add_argument("--limit_train", type=int, default=0, help="Pilot only: fixed class-stratified subset")
    p.add_argument("--limit_test", type=int, default=0, help="Pilot only: fixed class-stratified subset")
    p.add_argument("--skip_mia", action="store_true")
    p.add_argument("--pilot", action="store_true", help="Explicit pipeline/throughput check; excluded from scientific results")
    args = p.parse_args()
    if not 0 < args.select_ratio <= 1 or not 0 < args.retain_ratio <= 1:
        p.error("select_ratio and retain_ratio must lie in (0,1]")
    if not 0 <= args.client_id < args.num_clients:
        p.error("client_id is outside the federation")
    for key in ("num_clients", "global_epochs", "local_epochs", "batch_size", "threads"):
        if getattr(args, key) <= 0:
            p.error(f"{key} must be positive")
    if args.mode != "pretrain" and not args.checkpoint:
        p.error("All revision methods require --checkpoint from a shared pretrain run")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def json_save(path, value):
    # Atomic replace preserves the preceding checkpoint/report if interrupted.
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_checkpoint(path, model, metadata, partitions, schedule):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save({"model_state": state_cpu(model), "metadata": metadata,
                "partitions": partitions, "schedule": schedule}, temporary)
    temporary.replace(path)


def make_model(args):
    constructor = getattr(models, args.model)
    return constructor(weights=None, num_classes=10 if args.data == "cifar10" else 100)


def stratified_limit(targets, limit, seed):
    if not limit or limit >= len(targets):
        return np.arange(len(targets))
    rng = np.random.default_rng(seed)
    classes = np.unique(targets)
    chunks = []
    for index, cls in enumerate(classes):
        pool = np.flatnonzero(targets == cls)
        count = limit // len(classes) + int(index < limit % len(classes))
        chunks.extend(rng.permutation(pool)[:count].tolist())
    return np.array(chunks, dtype=int)


def load_data(args):
    cls = datasets.CIFAR10 if args.data == "cifar10" else datasets.CIFAR100
    training = cls(args.data_root, train=True, download=True)
    testing = cls(args.data_root, train=False, download=True)
    train_ids = stratified_limit(np.array(training.targets), args.limit_train, args.split_seed)
    test_ids = stratified_limit(np.array(testing.targets), args.limit_test, args.split_seed)
    def tensors(dataset, ids):
        # Same [-1,1] normalization as the supplied implementation. Data are
        # cached on CPU once, avoiding per-example PIL transforms in every round.
        images = torch.from_numpy(dataset.data[ids]).permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
        labels = torch.tensor(np.array(dataset.targets)[ids], dtype=torch.long)
        return TensorDataset(images, labels)
    return tensors(training, train_ids), tensors(testing, test_ids), train_ids.tolist(), test_ids.tolist()


def partition_labels(labels, args):
    rng = np.random.default_rng(args.split_seed)
    clients = [[] for _ in range(args.num_clients)]
    for cls in np.unique(labels):
        pool = rng.permutation(np.flatnonzero(labels == cls))
        if args.iid:
            chunks = np.array_split(pool, args.num_clients)
        else:
            proportions = rng.dirichlet(np.full(args.num_clients, args.dirichlet_alpha))
            counts = (proportions * len(pool)).astype(int)
            # Assign rounding residuals to the largest fractional remainders.
            residual = len(pool) - counts.sum()
            remainder_order = np.argsort(-(proportions * len(pool) - counts), kind="stable")
            counts[remainder_order[:residual]] += 1
            chunks = np.split(pool, np.cumsum(counts)[:-1])
        for client, chunk in zip(clients, chunks):
            client.extend(int(i) for i in chunk)
    if any(not client for client in clients):
        raise ValueError("Empty client partition; increase data or Dirichlet alpha")
    return clients


def retained_partitions(partitions, labels, args):
    affected = set(range(args.num_clients)) if args.request_scope == "global_class" else {args.client_id}
    target = set(args.unlearn_class)
    return [[i for i in indices if cid not in affected or int(labels[i]) not in target]
            for cid, indices in enumerate(partitions)]


def aggregate_states(states, counts, policy="samples"):
    if not states or len(states) != len(counts) or min(counts) <= 0:
        raise ValueError("Aggregation requires a nonempty list of positive-size clients")
    weights = np.array(counts, dtype=float) if policy == "samples" else np.ones(len(counts))
    weights /= weights.sum()
    result = {}
    for key in states[0]:
        if states[0][key].is_floating_point():
            result[key] = sum(state[key] * float(weight) for state, weight in zip(states, weights))
        else:
            # BatchNorm counters are integer state, not model parameters. Use
            # the maximum counter; never average and silently truncate it.
            result[key] = torch.stack([state[key] for state in states]).amax(0)
    return result


def loader(dataset, indices, args, shuffle=False, seed=0):
    selected = dataset if indices is None else Subset(dataset, list(indices))
    return DataLoader(selected, batch_size=args.batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed),
                      pin_memory=str(args.resolved_device).startswith("cuda"), num_workers=0)


def train_epochs(model, dataset, indices, args, epochs, lr, seed, ascent=False):
    if not indices:
        return 0
    batches = loader(dataset, indices, args, shuffle=True, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    steps = 0
    model.train()
    for _ in range(epochs):
        for x, y in batches:
            # The 32x32 standard ResNet ends in a 1x1 feature map. A singleton
            # final minibatch cannot estimate BatchNorm variance; use stored
            # statistics for that minibatch, keeping its example and gradients.
            model.train(len(x) > 1)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x.to(args.resolved_device)), y.to(args.resolved_device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is not finite")
            (-loss if ascent else loss).backward()
            optimizer.step()
            steps += 1
    return steps


def synchronize(args):
    if args.resolved_device.startswith("cuda"):
        torch.cuda.synchronize()


def federated_train(model, train, partitions, schedule, args, run_dir, metadata):
    start = time.perf_counter()
    steps = 0
    for round_index, client_ids in enumerate(schedule):
        states, counts = [], []
        for cid in client_ids:
            if not partitions[cid]:
                continue
            local = copy.deepcopy(model)
            steps += train_epochs(local, train, partitions[cid], args, args.local_epochs,
                                  args.lr, args.seed + round_index * args.num_clients + cid)
            states.append(state_cpu(local))
            counts.append(len(partitions[cid]))
            del local
        if states:
            model.load_state_dict(aggregate_states(states, counts, args.aggregation))
        synchronize(args)
        json_save(run_dir / "progress.json", {"round": round_index + 1,
                    "total_rounds": len(schedule), "steps": steps,
                    "elapsed_seconds": time.perf_counter() - start})
        print(f"round {round_index+1}/{len(schedule)} steps={steps} elapsed={time.perf_counter()-start:.1f}s", flush=True)
        save_checkpoint(run_dir / "latest_model.pt", model, metadata, partitions, schedule)
    return steps


def optimized_noise(model, target, args):
    was_training = model.training
    flags = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    noise = nn.Parameter(torch.randn(args.batch_size, 3, 32, 32, device=args.resolved_device))
    optimizer = torch.optim.Adam([noise], lr=args.noise_lr)
    labels = torch.full((args.batch_size,), target, device=args.resolved_device, dtype=torch.long)
    try:
        for _ in range(args.noise_epochs * args.noise_steps):
            optimizer.zero_grad(set_to_none=True)
            penalty = (noise.square().sum((1, 2, 3)).mean() if args.mode == "unsir"
                       else torch.linalg.vector_norm(noise))
            loss = -nn.functional.cross_entropy(model(noise), labels) + args.noise_l2 * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError("Noise objective is not finite")
            loss.backward()
            optimizer.step()
    finally:
        for parameter, flag in zip(model.parameters(), flags):
            parameter.requires_grad_(flag)
        model.train(was_training)
    return noise.detach().cpu()


def select_retained(model, train, indices, args, cid):
    policy = args.selection if args.selection != "auto" else ("random" if args.iid or args.mode == "unsir" else "confidence")
    labels = train.tensors[1].numpy()
    rng = np.random.default_rng(args.unlearn_seed + cid)
    scores = None
    if policy == "confidence":
        p, _ = posterior(model, loader(train, indices, args), args.resolved_device)
        scores = dict(zip(indices, p.max(1)))
    selected = []
    for cls in np.unique(labels[indices]):
        pool = [i for i in indices if labels[i] == cls]
        count = max(1, int(len(pool) * args.retain_ratio))
        if policy == "random":
            pool = rng.permutation(pool).tolist()
        else:
            pool = sorted(pool, key=lambda i: (-scores[i], i))
        selected.extend(pool[:count])
    return selected


def unlearn(model, train, partitions, retain, args, run_dir):
    affected = list(range(args.num_clients)) if args.request_scope == "global_class" else [args.client_id]
    labels = train.tensors[1].numpy()
    target_set = set(args.unlearn_class)
    affected = [cid for cid in affected if any(int(labels[i]) in target_set for i in partitions[cid])]
    if not affected:
        raise ValueError("No target examples in the request scope")
    noises = {}
    if args.mode in ("noise", "unsir") and args.ablation != "repair_only":
        noises = {target: optimized_noise(model, target, args) for target in args.unlearn_class}
    states, counts, selection_record = [], [], {}
    steps = 0
    for cid in affected:
        local = copy.deepcopy(model)
        if args.mode in ("noise", "unsir"):
            if not retain[cid]:
                raise ValueError(f"Client {cid} has no retained examples for impair/repair")
            selected = select_retained(model, train, retain[cid], args, cid)
            selection_record[str(cid)] = {"indices": selected, "selected_count": len(selected),
                "full_retain_count": len(retain[cid])}
            if args.ablation != "repair_only":
                for target, noise in noises.items():
                    images = noise.repeat(args.noise_copies, 1, 1, 1)
                    noise_data = TensorDataset(images, torch.full((len(images),), target, dtype=torch.long))
                    mixed = ConcatDataset([noise_data, Subset(train, selected)])
                    steps += train_epochs(local, mixed, list(range(len(mixed))), args, 1,
                                 args.impair_lr, args.unlearn_seed + cid)
            if args.ablation != "impair_only":
                steps += train_epochs(local, train, selected, args, args.repair_epochs,
                             args.repair_lr, args.unlearn_seed + cid)
        elif args.mode == "ft":
            steps += train_epochs(local, train, retain[cid], args, args.ft_epochs,
                         args.ft_lr, args.unlearn_seed + cid)
        elif args.mode == "ga":
            forgotten = [i for i in partitions[cid] if int(labels[i]) in target_set]
            steps += train_epochs(local, train, forgotten, args, args.ga_epochs,
                         args.ga_lr, args.unlearn_seed + cid, ascent=True)
        states.append(state_cpu(local))
        counts.append(len(retain[cid]))
        del local
        print(f"{args.mode}: client {cid} completed", flush=True)
    model.load_state_dict(aggregate_states(states, counts, args.aggregation))
    json_save(run_dir / "selection.json", selection_record)
    model_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    noise_bytes = sum(t.numel() * t.element_size() for t in noises.values())
    return {"participating_clients": affected, "model_download_bytes": model_bytes * len(affected),
            "noise_download_bytes": noise_bytes * len(affected), "model_upload_bytes": model_bytes * len(affected),
            "network_latency_measured": False}, steps


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    args.resolved_device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    seed_everything(args.seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = Path(args.output_dir).resolve() / f"{stamp}_{args.mode}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    sys.stdout = Tee(sys.stdout, run_dir / "run.log")
    sys.stderr = Tee(sys.stderr, run_dir / "errors.log")
    previous_hook = sys.excepthook
    def failure_hook(kind, value, trace):
        json_save(run_dir / "status.json", {"status": "failed", "exception": kind.__name__, "message": str(value)})
        previous_hook(kind, value, trace)
    sys.excepthook = failure_hook
    for name in ("revision_runner.py", "revision_evaluation.py"):
        shutil.copy2(Path(__file__).with_name(name), run_dir / name)
    config = vars(args) | {"started_utc": stamp, "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.resolved_device) if args.resolved_device.startswith("cuda") else None,
        "pilot": bool(args.pilot or args.limit_train or args.limit_test), "runner_schema": 2,
        "runner_sha256": file_hash(__file__),
        "evaluation_sha256": file_hash(Path(__file__).with_name("revision_evaluation.py"))}
    json_save(run_dir / "config.json", config)
    json_save(run_dir / "status.json", {"status": "running", "phase": "data"})
    print(f"RUN_DIR={run_dir}", flush=True)
    train, test, train_ids, test_ids = load_data(args)
    labels = train.tensors[1].numpy()
    metadata = copy.deepcopy(config)
    metadata["data_indices"] = {"train": train_ids, "test": test_ids}
    loaded = None
    if args.checkpoint:
        loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        source = loaded["metadata"]
        for key in ("model", "data", "num_clients", "select_ratio", "global_epochs", "local_epochs", "batch_size", "lr", "seed", "split_seed",
                    "iid", "dirichlet_alpha", "aggregation", "limit_train", "limit_test"):
            if source.get(key) != getattr(args, key):
                raise ValueError(f"Shared checkpoint mismatch: {key}: {source.get(key)} != {getattr(args,key)}")
        if source["data_indices"] != metadata["data_indices"]:
            raise ValueError("Evaluation/training sample IDs differ from shared checkpoint")
        partitions, schedule = loaded["partitions"], loaded["schedule"]
        config["checkpoint_sha256"] = file_hash(args.checkpoint)
        metadata["checkpoint_sha256"] = config["checkpoint_sha256"]
    else:
        partitions = partition_labels(labels, args)
        rng = random.Random(args.seed)
        selected = max(1, int(args.select_ratio * args.num_clients))
        schedule = [rng.sample(range(args.num_clients), selected) for _ in range(args.global_epochs)]
    flat = [i for part in partitions for i in part]
    if sorted(flat) != list(range(len(train))):
        raise ValueError("Partitions must assign every training sample exactly once")
    retain = retained_partitions(partitions, labels, args)
    json_save(run_dir / "partitions.json", partitions)
    json_save(run_dir / "retained_partitions.json", retain)
    json_save(run_dir / "client_schedule.json", schedule)
    json_save(run_dir / "data_indices.json", metadata["data_indices"])
    json_save(run_dir / "config.json", config)
    model = make_model(args).to(args.resolved_device)
    if loaded:
        model.load_state_dict(loaded["model_state"])
    before_p, test_y = posterior(model, loader(test, None, args), args.resolved_device)
    before = accuracy_metrics(before_p, test_y, args.unlearn_class)
    method_seed = args.seed if args.mode == "pretrain" else (args.retrain_seed if args.mode == "retrain" else args.unlearn_seed)
    seed_everything(method_seed)
    synchronize(args)
    start = time.perf_counter()
    communication, steps = {}, None
    if args.mode in ("pretrain", "retrain"):
        if args.mode == "retrain":
            del model
            model = make_model(args).to(args.resolved_device)
        steps = federated_train(model, train, retain if args.mode == "retrain" else partitions,
                                schedule, args, run_dir, metadata)
    else:
        communication, steps = unlearn(model, train, partitions, retain, args, run_dir)
    # Timing includes state serialization, excludes post-hoc evaluation and network latency.
    save_checkpoint(run_dir / "final_model.pt", model, metadata, partitions, schedule)
    synchronize(args)
    elapsed = time.perf_counter() - start
    final_p, test_y = posterior(model, loader(test, None, args), args.resolved_device)
    results = {"before": before, "after": accuracy_metrics(final_p, test_y, args.unlearn_class),
               "method_seconds": elapsed, "optimizer_steps": steps, "communication": communication,
               "pilot": config["pilot"], "run_dir": str(run_dir)}
    json_save(run_dir / "status.json", {"status": "running", "phase": "evaluation"})
    np.savez_compressed(run_dir / "test_predictions.npz", probabilities=final_p, labels=test_y)
    train_p = None
    splits = None
    if not args.skip_mia:
        train_p, train_y = posterior(model, loader(train, None, args), args.resolved_device)
        member_indices = flat if args.request_scope == "global_class" else partitions[args.client_id]
        splits = make_attack_splits(train_y, test_y, member_indices, args.attack_seed)
        json_save(run_dir / "attack_splits.json", splits)
        results["membership"] = membership_diagnostics(train_p, train_y, final_p, test_y,
                                                       splits, args.unlearn_class)
    if args.reference:
        ref = torch.load(args.reference, map_location="cpu", weights_only=False)
        if ref["metadata"]["mode"] != "retrain":
            raise ValueError("--reference must be a retrain checkpoint")
        for key in ("request_scope", "unlearn_class", "client_id", "attack_seed", "model", "data", "split_seed", "seed", "retrain_seed"):
            if ref["metadata"][key] != getattr(args, key):
                raise ValueError(f"Reference protocol mismatch: {key}")
        if ref["partitions"] != partitions or ref["metadata"]["data_indices"] != metadata["data_indices"]:
            raise ValueError("Reference partitions or sample IDs differ")
        reference = make_model(args).to(args.resolved_device)
        reference.load_state_dict(ref["model_state"])
        reference_p, _ = posterior(reference, loader(test, None, args), args.resolved_device)
        results["retrain_distance"] = prediction_distance(final_p, reference_p)
        if train_p is not None:
            reference_train_p, _ = posterior(reference, loader(train, None, args), args.resolved_device)
            reference_mia = membership_diagnostics(reference_train_p, train_y, reference_p,
                                                   test_y, splits, args.unlearn_class)
            results["reference_membership"] = reference_mia
            results["membership_difference_from_retrain"] = {
                group: {key: results["membership"]["groups"][group][key] - reference_mia["groups"][group][key]
                        for key in ("macro_auc", "macro_advantage")
                        if results["membership"]["groups"][group][key] is not None}
                for group in ("target", "retained")}
        results["reference_sha256"] = file_hash(args.reference)
    json_save(run_dir / "metrics.json", results)
    json_save(run_dir / "status.json", {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
