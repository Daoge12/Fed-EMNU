"""Audit and summarize the completed frozen suite; never import pilot results."""
import argparse
import csv
import hashlib
import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LABELS = {"pretrain": "Original", "retrain": "Retrain", "ft": "Finetune", "ga": "GA",
          "noise": "Fed-EMNU", "unsir": "UNSIR-style", "selection_random": "Random (rt=0.5)",
          "selection_confidence": "Confidence (rt=0.5)", "impair_only": "Impair only",
          "repair_only": "Repair only"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite scientific measurement")
    n = len(data)
    return {"n": n, "mean": float(data.mean()),
            "sd": float(data.std(ddof=1)) if n > 1 else None}


def read_runs(suite):
    if suite["spec"]["pilot"]:
        raise ValueError("Pilot suites cannot be scientific evidence")
    dataset = suite['spec'].get('extension', {}).get('data', 'cifar10')
    class_count = 100 if dataset == 'cifar100' else 10
    rows, evidence, missing = [], [], []
    for key, run in suite["runs"].items():
        if run["status"] != "complete":
            missing.append(key)
            continue
        directory = Path(run["run_dir"])
        config = json.loads((directory / "config.json").read_text())
        metrics = json.loads((directory / "metrics.json").read_text())
        status = json.loads((directory / "status.json").read_text())
        if status["status"] != "complete" or config["pilot"] or metrics["pilot"]:
            raise ValueError(f"Invalid scientific completion: {key}")
        if config["limit_train"] or config["limit_test"] or config["global_epochs"] != 50 or config["local_epochs"] != 5:
            raise ValueError(f"Full fixed-budget protocol required: {key}")
        for name, expected in suite["spec"]["source_hashes"].items():
            if digest(directory / name) != expected:
                raise ValueError(f"Source snapshot mismatch: {key}/{name}")
        if config["request_scope"] != "global_class" or config["unlearn_class"] != [0]:
            raise ValueError("This summary is specific to global class 0")
        predictions = np.load(directory / "test_predictions.npz")
        probabilities, labels = predictions["probabilities"], predictions["labels"]
        if config['data'] != dataset:
            raise ValueError('Suite and run dataset mismatch')
        if probabilities.shape != (10000, class_count) or not np.isfinite(probabilities).all():
            raise ValueError(f"Invalid test prediction artifact: {key}")
        correct = probabilities.argmax(1) == labels
        target = labels == 0
        actual = {"test_acc": correct.mean(), "forget_acc": correct[target].mean(),
                  "retain_acc": correct[~target].mean()}
        if not all(np.isclose(actual[metric], metrics["after"][metric], atol=1e-12) for metric in actual):
            raise ValueError(f"Saved predictions do not reproduce accuracy: {key}")
        if not np.allclose(probabilities.sum(1), 1, atol=1e-5):
            raise ValueError(f"Posterior rows are not normalized: {key}")
        splits = json.loads((directory / "attack_splits.json").read_text())
        if set(splits) != {str(c) for c in range(class_count)}:
            raise ValueError('Missing attack classes in full global-class evaluation')
        for cls, split in splits.items():
            if set(split["member_calibration"]) & set(split["member_test"]):
                raise ValueError("Attack member calibration/test overlap")
            if set(split["nonmember_calibration"]) & set(split["nonmember_test"]):
                raise ValueError("Attack nonmember calibration/test overlap")
            if len(split["member_test"]) != len(split["nonmember_test"]):
                raise ValueError("Unbalanced within-class attack test groups")
            if any(labels[i] != int(cls) for i in split["nonmember_test"] + split["nonmember_calibration"]):
                raise ValueError("Attack nonmember class mismatch")
        setting, _, remainder = key.partition("_seed")
        seed, _, mode = remainder.partition("_")
        row = {"setting": setting, "seed": int(seed), "method": mode,
               "fa": metrics["after"]["forget_acc"], "ra": metrics["after"]["retain_acc"],
               "accuracy": metrics["after"]["test_acc"], "seconds": metrics["method_seconds"],
               "local_optimizer_steps": metrics["optimizer_steps"]}
        if config["mode"] == "retrain":
            row.update(disagreement=0.0, tv=0.0)
        elif "retrain_distance" in metrics:
            row.update(disagreement=metrics["retrain_distance"]["prediction_disagreement"],
                       tv=metrics["retrain_distance"]["total_variation"])
        for group in ("target", "retained"):
            for metric in ("macro_auc", "macro_advantage"):
                row[group + "_" + metric] = metrics["membership"]["groups"][group][metric]
                if "membership_difference_from_retrain" in metrics:
                    row["delta_" + group + "_" + metric] = metrics["membership_difference_from_retrain"][group][metric]
        communication = metrics.get("communication", {})
        if "model_download_bytes" in communication:
            row["event_payload_bytes"] = sum(communication.get(name, 0) for name in
                ("model_download_bytes", "noise_download_bytes", "model_upload_bytes"))
        rows.append(row)
        evidence.append({"key": key, "run_dir": str(directory),
                         "config_sha256": digest(directory / "config.json"),
                         "metrics_sha256": digest(directory / "metrics.json"),
                         "predictions_sha256": digest(directory / "test_predictions.npz"),
                         "checkpoint_sha256": digest(directory / "final_model.pt")})
    # Verify all completed comparisons used the same before-update model behavior.
    for setting in suite["spec"]["settings"]:
        for seed in suite["spec"]["seeds"]:
            names = [e for e in evidence if e["key"].startswith(f"{setting}_seed{seed}_")]
            original = next((e for e in names if e["key"].endswith("_pretrain")), None)
            if original:
                original_metrics = json.loads((Path(original["run_dir"]) / "metrics.json").read_text())
                original_config = json.loads((Path(original["run_dir"]) / "config.json").read_text())
                for item in names:
                    if item == original:
                        continue
                    config = json.loads((Path(item["run_dir"]) / "config.json").read_text())
                    metrics = json.loads((Path(item["run_dir"]) / "metrics.json").read_text())
                    if config["checkpoint_sha256"] != original["checkpoint_sha256"] or metrics["before"] != original_metrics["after"]:
                        raise ValueError("Shared checkpoint/before metrics mismatch")
                    for field in ("model", "data", "iid", "aggregation", "split_seed", "select_ratio"):
                        if config[field] != original_config[field]:
                            raise ValueError(f"Protocol mismatch: {field}")
                    reference = next((e for e in names if e["key"].endswith("_retrain")), None)
                    if reference and "reference_sha256" in metrics:
                        if metrics["reference_sha256"] != reference["checkpoint_sha256"]:
                            raise ValueError("Wrong retraining checkpoint hash")
                        actual_p = np.load(Path(item["run_dir"]) / "test_predictions.npz")["probabilities"]
                        reference_p = np.load(Path(reference["run_dir"]) / "test_predictions.npz")["probabilities"]
                        measured = {"prediction_disagreement": float((actual_p.argmax(1) != reference_p.argmax(1)).mean()),
                                    "total_variation": float((0.5 * np.abs(actual_p-reference_p).sum(1)).mean())}
                        if any(not np.isclose(value, metrics["retrain_distance"][metric], atol=1e-10)
                               for metric, value in measured.items()):
                            raise ValueError("Saved predictions do not reproduce reference distance")
    label_chunks = []
    if dataset == 'cifar100':
        with (ROOT / 'code/data/cifar-100-python/train').open('rb') as stream:
            label_chunks.extend(pickle.load(stream, encoding='bytes')[b'fine_labels'])
    else:
        for index in range(1, 6):
            with (ROOT / f"code/data/cifar-10-batches-py/data_batch_{index}").open("rb") as stream:
                label_chunks.extend(pickle.load(stream, encoding="bytes")[b"labels"])
    training_labels = np.asarray(label_chunks)
    for item in evidence:
        directory = Path(item["run_dir"])
        metrics = json.loads((directory / "metrics.json").read_text())
        splits = json.loads((directory / "attack_splits.json").read_text())
        for cls, split in splits.items():
            member_ids = split["member_calibration"] + split["member_test"]
            if len(member_ids) != len(set(member_ids)) or any(training_labels[i] != int(cls) for i in member_ids):
                raise ValueError("Attack member IDs are duplicated or from another class")
        for attack_name in ("membership", "reference_membership"):
            if attack_name not in metrics:
                continue
            attack = metrics[attack_name]
            for group in ("target", "retained"):
                class_rows = [record for cls, record in attack["per_class"].items()
                              if (int(cls) == 0) == (group == "target")]
                for metric in ("auc", "advantage"):
                    measured = np.mean([record[metric] for record in class_rows])
                    if not np.isclose(measured, attack["groups"][group]["macro_" + metric], atol=1e-12):
                        raise ValueError("Per-class attacks do not reproduce macro metric")
        if "reference_membership" in metrics:
            for group in ("target", "retained"):
                for metric in ("macro_auc", "macro_advantage"):
                    measured = metrics["membership"]["groups"][group][metric] - metrics["reference_membership"]["groups"][group][metric]
                    if not np.isclose(measured, metrics["membership_difference_from_retrain"][group][metric], atol=1e-12):
                        raise ValueError("Attack difference does not match method/reference metrics")
        selection_path = directory / "selection.json"
        if not selection_path.exists():
            continue
        selection = json.loads(selection_path.read_text())
        retain = json.loads((directory / "retained_partitions.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        for client, record in selection.items():
            indices = record["indices"]
            if len(indices) != len(set(indices)) or not set(indices) <= set(retain[int(client)]):
                raise ValueError("Selection contains duplicate or inaccessible records")
            pool_labels = training_labels[retain[int(client)]]
            selected_labels = training_labels[indices]
            for cls in np.unique(pool_labels):
                expected = max(1, int((pool_labels == cls).sum() * config["retain_ratio"]))
                if (selected_labels == cls).sum() != expected:
                    raise ValueError("Selected within-class count differs from declared ratio")
            if np.isin(selected_labels, config["unlearn_class"]).any():
                raise ValueError("Retained selection includes revoked class")
    return rows, evidence, missing


def aggregate(rows):
    groups = {}
    for row in rows:
        key = row["setting"] + "/" + row["method"]
        group = groups.setdefault(key, {})
        for metric, value in row.items():
            if metric not in ("setting", "seed", "method"):
                group.setdefault(metric, []).append(value)
    return {key: {metric: stats(values) for metric, values in group.items()} for key, group in groups.items()}


def paired(rows):
    results = {}
    for setting in ("iid", "noniid"):
        random = {r["seed"]: r for r in rows if r["setting"] == setting and r["method"] == "selection_random"}
        confidence = {r["seed"]: r for r in rows if r["setting"] == setting and r["method"] == "selection_confidence"}
        seeds = sorted(random.keys() & confidence.keys())
        if seeds:
            results[setting] = {"seeds": seeds, "confidence_minus_random": {
                metric: stats([confidence[s][metric] - random[s][metric] for s in seeds])
                for metric in ("fa", "ra", "accuracy", "seconds", "tv", "target_macro_auc")}}
    return results


def speedups(rows):
    result = {}
    for setting in ("iid", "noniid"):
        times = {(r["seed"], r["method"]): r["seconds"] for r in rows if r["setting"] == setting}
        seeds = sorted({seed for seed, mode in times if mode == "noise"})
        for baseline in ("retrain", "ft", "ga", "unsir"):
            ratios = [times[s, baseline] / times[s, "noise"] for s in seeds if (s, baseline) in times]
            if ratios:
                result[setting + "/" + baseline + "_over_emnu"] = stats(ratios)
    return result


def reference_variation(evidence):
    result = {}
    for setting in ("iid", "noniid"):
        refs = [item for item in evidence if item["key"].startswith(setting + "_") and item["key"].endswith("_retrain")]
        pairs = []
        for left, right in combinations(refs, 2):
            a = np.load(Path(left["run_dir"]) / "test_predictions.npz")
            b = np.load(Path(right["run_dir"]) / "test_predictions.npz")
            if not np.array_equal(a["labels"], b["labels"]):
                raise ValueError("Reference-variation test sets differ")
            p, q = a["probabilities"], b["probabilities"]
            pairs.append({"left": left["key"], "right": right["key"],
                          "disagreement": float((p.argmax(1) != q.argmax(1)).mean()),
                          "tv": float((.5*np.abs(p-q).sum(1)).mean())})
        if pairs:
            result[setting] = {"pairs": pairs, "independent_pair_samples": False,
                               "note": "Partition, schedule and initialization all vary across run seeds; these dependent pairs are descriptive, not a same-partition distributional test."}
    return result


def formatted(group, metric, percent=False):
    entry = group[metric]
    scale = 100 if percent else 1
    digits = 2 if percent or metric == "seconds" else 3
    mean = f"{entry['mean'] * scale:.{digits}f}"
    sd = f"{entry['sd'] * scale:.{digits}f}" if entry["sd"] is not None else "NA"
    return f"${mean} \\pm {sd}$"


def latex(summary, suite, paired_rows, cost_rows, reference_rows):
    lines = [r"\noindent The following results were generated from completed full-budget runs of the frozen revision protocol. "
             r"Values are arithmetic mean $\pm$ sample SD across independent seeds; accuracy and disagreement use percentage points. "
             r"Retrain self-distance is zero by definition and does not measure variability between independent references."]
    core = ("pretrain", "retrain", "ft", "ga", "noise", "unsir")
    for setting in suite["spec"]["settings"]:
        name = "IID" if setting == "iid" else r"Non-IID ($\alpha=0.5$)"
        lines += [r"\subsection{" + name + " comparison}", r"\begin{table}[htbp]\centering\small",
                  r"\caption{" + name + r" utility and method time. Original time is pretraining cost; Retrain is the retained-data reference.}",
                  r"\label{tab:" + setting + r"-utility}",
                  r"\begin{tabular}{lrrrrr}\toprule Method & $n$ & FA (\%) & RA (\%) & Acc. (\%) & Time (s)\\\midrule"]
        for mode in core:
            group = summary[f"{setting}/{mode}"]
            lines += [LABELS[mode] + " & " + str(group["fa"]["n"]) + " & " + " & ".join(
                formatted(group, key, key != "seconds") for key in ("fa", "ra", "accuracy", "seconds")) + r"\\"]
        lines += [r"\bottomrule\end{tabular}\end{table}", r"\begin{table}[htbp]\centering\small",
                  r"\caption{" + name + r" reference distance and class-conditioned attack. AUC and advantage are target-class values.}",
                  r"\label{tab:" + setting + r"-distance}",
                  r"\begin{tabular}{lrrrr}\toprule Method & Disagree (\%) & TV & Target AUC & Target advantage\\\midrule"]
        for mode in core[1:]:
            group = summary[f"{setting}/{mode}"]
            lines += [LABELS[mode] + " & " + " & ".join(formatted(group, key, key == "disagreement") for key in
                ("disagreement", "tv", "target_macro_auc", "target_macro_advantage")) + r"\\"]
        lines += [r"\bottomrule\end{tabular}\end{table}"]
        lines += [r"\begin{table}[htbp]\centering\small",
                  r"\caption{" + name + r" retained-class attack and attack differences from Retrain. Differences use identical member/nonmember indices and separately calibrated model-specific thresholds.}",
                  r"\label{tab:" + setting + r"-attack}",
                  r"\begin{tabular}{lrrrr}\toprule Method & Retained AUC & Retained adv. & $\Delta$ target AUC & $\Delta$ retained AUC\\\midrule"]
        for mode in core[1:]:
            group = summary[f"{setting}/{mode}"]
            base = " & ".join(formatted(group, key) for key in ("retained_macro_auc", "retained_macro_advantage"))
            differences = "0 (self) & 0 (self)" if mode == "retrain" else " & ".join(formatted(group, key) for key in
                ("delta_target_macro_auc", "delta_retained_macro_auc"))
            lines += [LABELS[mode] + " & " + base + " & " + differences + r"\\"]
        lines += [r"\bottomrule\end{tabular}\end{table}"]
        if suite["spec"]["controls"]:
            lines += [r"\begin{table}[htbp]\centering\small",
                      r"\caption{" + name + r" matched selection and stage controls. All values use the same reference and pretrained states as the main comparison.}",
                      r"\label{tab:" + setting + r"-controls}",
                      r"\begin{tabular}{lrrrr}\toprule Control & FA (\%) & RA (\%) & TV & Time (s)\\\midrule"]
            for mode in ("selection_random", "selection_confidence", "impair_only", "repair_only"):
                group = summary[f"{setting}/{mode}"]
                lines += [LABELS[mode] + " & " + " & ".join(formatted(group, key, key in ("fa", "ra")) for key in
                    ("fa", "ra", "tv", "seconds")) + r"\\"]
            lines += [r"\bottomrule\end{tabular}\end{table}"]
        emnu = summary[f"{setting}/noise"]
        unsir = summary[f"{setting}/unsir"]
        ref = summary[f"{setting}/retrain"]
        lines += [r"\noindent\textbf{Behavioral and leakage interpretation.} " +
                  f"Fed-EMNU achieved FA {formatted(emnu, 'fa', True)}\\% and RA {formatted(emnu, 'ra', True)}\\%, "
                  f"whereas Retrain achieved FA {formatted(ref, 'fa', True)}\\%. "
                  f"The UNSIR-style adaptation achieved FA {formatted(unsir, 'fa', True)}\\% and RA {formatted(unsir, 'ra', True)}\\%. "
                  f"Fed-EMNU's mean posterior TV from Retrain was {formatted(emnu, 'tv')}; "
                  f"its target attack AUC was {formatted(emnu, 'target_macro_auc')} compared with "
                  f"{formatted(ref, 'target_macro_auc')} for Retrain. "
                  "These observations must be considered jointly: retained accuracy can be preserved while target behavior "
                  "and membership scores remain different from the retained-data reference. "
                  "The UNSIR-style comparison changes both the noise regularizer/budget and repair budget, "
                  "so its difference from Fed-EMNU cannot be assigned to one component alone."]
        lines += [""]
    lines += [r"\begin{figure}[htbp]\centering",
              r"\includegraphics[width=\linewidth]{figures/revision_core_metrics.pdf}",
              r"\caption{Full-budget utility, reference distance and runtime. Dots are independent seeds; bars show mean and sample SD. Runtime uses a logarithmic axis.}",
              r"\label{fig:core-results}",
              r"\end{figure}"]
    lines += [r"\subsection{Paired selection and cost comparisons}"]
    for setting, comparison in paired_rows.items():
        delta = comparison["confidence_minus_random"]
        ra = delta["ra"]
        fa = delta["fa"]
        lines += [f"For {setting.upper()}, confidence minus random selection changed RA by "
                  f"${100*ra['mean']:.2f} \\pm {100*ra['sd']:.2f}$ percentage points and FA by "
                  f"${100*fa['mean']:.2f} \\pm {100*fa['sd']:.2f}$ percentage points "
                  f"across {ra['n']} paired seeds. These small-sample comparisons do not establish a general ranking. "]
    for setting in suite["spec"]["settings"]:
        parts = []
        for baseline in ("retrain", "ft", "ga", "unsir"):
            entry = cost_rows[setting + "/" + baseline + "_over_emnu"]
            parts.append(f"{LABELS[baseline]}: ${entry['mean']:.2f} \\pm {entry['sd']:.2f}$")
        lines += [f"For {setting.upper()}, paired baseline time divided by Fed-EMNU time was " +
                  "; ".join(parts) + ". A ratio below one means the named baseline was faster. "
                  "These are compute/serialization ratios, excluding evaluation and real-network transport. "]
    lines += [r"\subsection{Stage controls and communication payload}"]
    for setting in suite["spec"]["settings"]:
        full = summary[f"{setting}/noise"]
        impair = summary[f"{setting}/impair_only"]
        repair = summary[f"{setting}/repair_only"]
        payload = full["event_payload_bytes"]
        lines += [f"For {setting.upper()}, impair-only RA was {formatted(impair, 'ra', True)}\\% "
                  f"and full-method RA was {formatted(full, 'ra', True)}\\%. "
                  f"Repair-only FA was {formatted(repair, 'fa', True)}\\%, compared with "
                  f"{formatted(full, 'fa', True)}\\% for the full method. "
                  "These controls identify changes associated with the prescribed stages in this protocol, "
                  "rather than proving removal of the target records. "
                  f"The full event transmitted {payload['mean']/2**20:.2f} MiB of tensor payload on average "
                  "(model downloads, noise and returned states combined). Network delay and framing "
                  "were not measured; compute-time reduction is not itself a communication guarantee."]
    lines += [r"\subsection{Variation among retraining references}"]
    for setting, comparison in reference_rows.items():
        pairs = comparison["pairs"]
        tv = [pair["tv"] for pair in pairs]
        disagreement = [100*pair["disagreement"] for pair in pairs]
        lines += [f"For {setting.upper()}, pairwise TV among the three retained-data references ranged from "
                  f"{min(tv):.3f} to {max(tv):.3f}, and prediction disagreement ranged from "
                  f"{min(disagreement):.2f}\\% to {max(disagreement):.2f}\\%. "
                  "These comparisons share the same retained global records and test set but vary initialization, "
                  "client partition and schedule together. The three pairs are dependent and are reported only "
                  "as descriptive context, not as additional independent runs or a certified tolerance for unlearning."]
    content = "\n".join(lines) + "\n"
    content = content.replace(r"\begin{table}[htbp]", r"\begin{table}[H]")
    content = content.replace("\\clearpage\n\\clearpage", "\\clearpage")
    return content.replace("For NONIID,", "For Non-IID,")


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    modes = ("retrain", "ft", "ga", "noise", "unsir")
    figure, axes = plt.subplots(3, 2, figsize=(8, 8.2), constrained_layout=True)
    for i, setting in enumerate(("iid", "noniid")):
        for j, (metric, title, scale) in enumerate((("ra", "Retained accuracy (%)", 100), ("tv", "TV to retraining", 1), ("seconds", "Method time (s)", 1))):
            ax = axes[j, i]
            for x, mode in enumerate(modes):
                values = np.array([r[metric] * scale for r in rows if r["setting"] == setting and r["method"] == mode])
                ax.errorbar(x, values.mean(), yerr=values.std(ddof=1), fmt="s", color="#124d75", capsize=4, zorder=3)
                ax.scatter(x + np.linspace(-.12, .12, len(values)), values, color="#ca6b27", s=19, zorder=4)
            ax.set_xticks(range(len(modes)), [LABELS[m] for m in modes], rotation=35, ha="right")
            ax.set_title(("IID" if i == 0 else "Non-IID") + ": " + title, fontsize=11)
            ax.grid(axis="y", alpha=.22)
            if metric == "seconds":
                ax.set_yscale("log")
            else:
                ax.set_ylim(bottom=0)
    figure.savefig(output / "revision_core_metrics.pdf", bbox_inches="tight")
    figure.savefig(output / "revision_core_metrics.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="experiments/core-suite/suite.json")
    parser.add_argument("--publish", action="store_true", help="Write manuscript tables/plot only if every planned run is complete")
    args = parser.parse_args()
    path = ROOT / args.suite
    suite = json.loads(path.read_text())
    modes = ["pretrain", "retrain", "ft", "ga", "noise", "unsir"]
    if suite["spec"]["controls"]:
        modes += ["selection_random", "selection_confidence", "impair_only", "repair_only"]
    expected = {f"{setting}_seed{seed}_{mode}" for setting in suite["spec"]["settings"]
                for seed in suite["spec"]["seeds"] for mode in modes}
    rows, evidence, missing = read_runs(suite)
    if not rows:
        print("No completed scientific runs yet")
        return
    summary = aggregate(rows)
    output = path.parent / "summary"
    output.mkdir(exist_ok=True)
    report = {"suite_sha256": digest(path), "suite_status": suite["status"],
              "completed_runs": len(rows), "planned_runs": len(expected),
              "not_yet_started": sorted(expected - suite["runs"].keys()), "unfinished_started_runs": missing,
              "summary": summary, "paired_selection": paired(rows), "paired_speedups": speedups(rows),
              "reference_variation": reference_variation(evidence), "evidence": evidence}
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    class_rows = []
    for item in evidence:
        metrics = json.loads((Path(item["run_dir"]) / "metrics.json").read_text())
        for cls, record in metrics["membership"]["per_class"].items():
            class_rows.append({"run": item["key"], "class": cls, "model": "method", **record})
        for cls, record in metrics.get("reference_membership", {}).get("per_class", {}).items():
            class_rows.append({"run": item["key"], "class": cls, "model": "retrain_reference", **record})
    if class_rows:
        with (output / "classwise_attacks.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(class_rows[0]))
            writer.writeheader()
            writer.writerows(class_rows)
    if args.publish:
        if suite['spec'].get('extension') or suite['spec']['seeds'] != [42, 43, 44]:
            raise ValueError('Core publication template is frozen to the original core; use a separate extension/combined generator')
        if suite["status"] != "complete" or missing or expected != suite["runs"].keys() or len(suite["spec"]["seeds"]) < 3:
            raise ValueError("Publishing requires the entire suite complete and at least three seeds")
        figures = ROOT / "manuscript/figures"
        plot(rows, figures)
        (ROOT / "manuscript/revision_results.tex").write_text(latex(summary, suite, paired(rows), speedups(rows), reference_variation(evidence)), encoding="utf-8")
    print(f"Audited {len(rows)} completed scientific runs; summary={output}")


if __name__ == "__main__":
    main()
