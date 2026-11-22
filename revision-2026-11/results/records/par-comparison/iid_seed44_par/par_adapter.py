"""PAR-code-based adaptation to the revision's shared 32x32 checkpoint.

Source inspected: https://github.com/lizouss/PAR commit
c587e11e56c2be71b99ed716f6045e5d92346c19. This is not an exact reproduction
of the published paper. See experiments/PAR_COMPARISON_PROTOCOL.md.
The frozen canonical runner/evaluator are left unchanged.
"""
import copy
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, Subset, TensorDataset

import revision_runner as runner

COMMIT = "c587e11e56c2be71b99ed716f6045e5d92346c19"


class PARGenerator(nn.Module):
    """Public code architecture for three-channel non-MNIST synthetic inputs."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 128 * 8 * 8), nn.Unflatten(1, (128, 8, 8)),
            nn.BatchNorm2d(128), nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.LeakyReLU(.2, inplace=True), nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.LeakyReLU(.2, inplace=True), nn.Conv2d(64, 3, 3, padding=1),
            nn.BatchNorm2d(3, affine=True))

    def forward(self, z):
        return self.layers(z)


def generate(model, target, args, directory):
    generator = PARGenerator().to(args.resolved_device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=2)
    # The source's loader reuses each latent batch for eight calls; training
    # and extraction share its counter. After 40 calls extraction uses a new z.
    flags = [p.requires_grad for p in model.parameters()]
    was_training = model.training
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    history = []
    try:
        for epoch in range(5):
            z = torch.randn(args.batch_size, 128, device=args.resolved_device)
            labels = torch.full((args.batch_size,), target, dtype=torch.long,
                                device=args.resolved_device)
            for _ in range(8):
                optimizer.zero_grad(set_to_none=True)
                inputs = generator(z)
                loss = -nn.functional.cross_entropy(model(inputs), labels)
                loss = loss + .01 * inputs.square().sum((1, 2, 3)).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("PAR generator objective is nonfinite")
                loss.backward()
                optimizer.step()
                history.append(float(loss.detach()))
            scheduler.step(float(loss.detach()))
        z = torch.randn(args.batch_size, 128, device=args.resolved_device)
        # Keep generator in train mode, as in the public extraction loop.
        with torch.no_grad():
            images = torch.cat([generator(z).cpu() for _ in range(8)])
        if not torch.isfinite(images).all():
            raise FloatingPointError("PAR synthetic inputs are nonfinite")
        torch.save({"generator_state": runner.state_cpu(generator),
                    "synthetic_inputs": images, "objective_history": history},
                   directory / f"par_generator_target{target}.pt")
        return images, history
    finally:
        for parameter, flag in zip(model.parameters(), flags):
            parameter.requires_grad_(flag)
        model.train(was_training)


def par_update(model, train, partitions, retain, args, directory):
    if args.request_scope != "global_class" or args.unlearn_class != [0]:
        raise ValueError("The frozen PAR comparison evaluates global class 0 only")
    if any(not indices for indices in retain):
        raise ValueError("PAR all-client restoration needs retained examples at every client")
    images, history = generate(model, 0, args, directory)
    synthetic = TensorDataset(images, torch.zeros(len(images), dtype=torch.long))
    steps = 0
    progress = []
    # PAR uses every client, including clients without target examples.
    # It aggregates after impair, then after each of ten repair rounds.
    for stage in range(11):
        states, counts = [], []
        for cid in range(args.num_clients):
            local = copy.deepcopy(model)
            if stage == 0:
                data = ConcatDataset([synthetic, Subset(train, retain[cid])])
                indices = list(range(len(data)))
            else:
                data, indices = train, retain[cid]
            steps += runner.train_epochs(local, data, indices, args, 1, .001,
                                         args.unlearn_seed + stage*args.num_clients + cid)
            states.append(runner.state_cpu(local))
            counts.append(len(retain[cid]))
            del local
        # Equal floating-state weights follow the public PAR code. Integer BN
        # counters use canonical max rather than float division/truncation.
        model.load_state_dict(runner.aggregate_states(states, counts, "equal"))
        progress.append({"stage": "impair" if stage == 0 else "repair",
                         "repair_round": stage, "local_optimizer_steps": steps})
        runner.json_save(directory / "par_progress.json", progress)
        print(f"par: completed {'impair' if stage == 0 else 'repair '+str(stage)+'/10'}", flush=True)
    model_bytes = sum(t.numel()*t.element_size() for t in model.state_dict().values())
    noise_bytes = images.numel()*images.element_size()
    runner.json_save(directory / "par_protocol.json", {
        "method": "PAR-code-based adaptation", "source_commit": COMMIT,
        "input_size": 32, "all_clients": list(range(args.num_clients)),
        "generator_steps": 40, "generator_lr": .001, "noise_l2": .01,
        "synthetic_count_per_target": len(images), "repair_rounds": 10,
        "local_epochs_per_stage": 1, "local_lr": .001,
        "event_aggregation": "equal floating states; max integer BN counters",
        "checkpoint_selection": "final fixed round; no test selection",
        "objective_history": history})
    for name in ("par_adapter.py", "PAR_COMPARISON_PROTOCOL.md"):
        source = Path(__file__).with_name(name) if name.endswith(".py") else Path(__file__).resolve().parent.parent / "experiments" / name
        (directory / name).write_bytes(source.read_bytes())
    runner.json_save(directory / "par_source_hashes.json", {
        name: runner.file_hash(directory / name) for name in
        ("par_adapter.py", "PAR_COMPARISON_PROTOCOL.md", "revision_runner.py", "revision_evaluation.py")})
    return {"participating_clients": list(range(args.num_clients)),
            "model_download_bytes": model_bytes * args.num_clients * 11,
            "model_upload_bytes": model_bytes * args.num_clients * 11,
            "noise_download_bytes": noise_bytes * args.num_clients,
            "network_latency_measured": False}, steps


if __name__ == "__main__":
    # Reuse the full evaluation/provenance path while replacing only the event.
    # CLI mode is noise for schema compatibility; par_protocol.json identifies
    # the actual method and audits must check this marker, not infer from mode.
    runner.unlearn = par_update
    runner.main()
