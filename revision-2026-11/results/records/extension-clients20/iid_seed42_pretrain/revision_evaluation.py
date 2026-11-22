"""Behavioral and black-box membership diagnostics; no privacy certificate."""
import numpy as np
import torch


@torch.inference_mode()
def posterior(model, loader, device):
    model.eval()
    outputs, labels = [], []
    for x, y in loader:
        probabilities = model(x.to(device)).softmax(1)
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("Evaluation produced nonfinite posterior probabilities")
        outputs.append(probabilities.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(outputs), np.concatenate(labels)


def accuracy_metrics(probabilities, labels, target_classes):
    correct = probabilities.argmax(1) == labels
    target = np.isin(labels, target_classes)
    return {
        "test_acc": float(correct.mean()),
        "forget_acc": float(correct[target].mean()) if target.any() else None,
        "retain_acc": float(correct[~target].mean()) if (~target).any() else None,
        "test_count": int(len(labels)), "forget_count": int(target.sum()),
        "retain_count": int((~target).sum()),
    }


def prediction_distance(probabilities, reference):
    if probabilities.shape != reference.shape:
        raise ValueError("Prediction arrays must refer to the identical evaluation set")
    return {
        "prediction_disagreement": float((probabilities.argmax(1) != reference.argmax(1)).mean()),
        "total_variation": float((0.5 * np.abs(probabilities - reference).sum(1)).mean()),
    }


def auc(member_scores, nonmember_scores):
    """Mann-Whitney AUC with average ranks for ties."""
    scores = np.concatenate([member_scores, nonmember_scores])
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    n, m = len(member_scores), len(nonmember_scores)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * m))


def calibrate_threshold(member, nonmember):
    """Select orientation and threshold using calibration examples only."""
    best = (-1.0, 1, float("inf"))
    for sign in (1, -1):
        m, n = np.sort(sign * member), np.sort(sign * nonmember)
        thresholds = np.r_[-np.inf, np.unique(np.r_[m, n]), np.inf]
        tpr = 1 - np.searchsorted(m, thresholds, side="left") / len(m)
        fpr = 1 - np.searchsorted(n, thresholds, side="left") / len(n)
        index = int(np.argmax(tpr - fpr))
        candidate = (float((tpr - fpr)[index]), sign, float(thresholds[index]))
        if candidate[0] > best[0]:
            best = candidate
    return best[1], best[2]


def make_attack_splits(train_labels, test_labels, member_indices, seed=90210):
    """Balance membership separately within each class; save exact indices."""
    rng = np.random.default_rng(seed)
    member_indices = np.asarray(member_indices, dtype=int)
    result = {}
    for cls in np.unique(train_labels):
        members = member_indices[train_labels[member_indices] == cls].copy()
        nonmembers = np.flatnonzero(test_labels == cls)
        rng.shuffle(members)
        rng.shuffle(nonmembers)
        count = min(len(members), len(nonmembers))
        if count < 4:
            continue
        midpoint = count // 2
        result[str(int(cls))] = {
            "member_calibration": members[:midpoint].tolist(),
            "member_test": members[midpoint:count].tolist(),
            "nonmember_calibration": nonmembers[:midpoint].tolist(),
            "nonmember_test": nonmembers[midpoint:count].tolist(),
        }
    return result


def membership_diagnostics(train_p, train_y, test_p, test_y, splits, target_classes):
    # Negative true-label cross entropy is log true-label probability. Its
    # direction may reverse after unlearning; direction is calibrated, not
    # selected from held-out attack test performance.
    train_scores = np.log(np.maximum(train_p[np.arange(len(train_y)), train_y], 1e-30))
    test_scores = np.log(np.maximum(test_p[np.arange(len(test_y)), test_y], 1e-30))
    per_class = {}
    for cls, split in splits.items():
        mc = train_scores[split["member_calibration"]]
        nc = test_scores[split["nonmember_calibration"]]
        mt = train_scores[split["member_test"]]
        nt = test_scores[split["nonmember_test"]]
        sign, threshold = calibrate_threshold(mc, nc)
        tpr = float((sign * mt >= threshold).mean())
        fpr = float((sign * nt >= threshold).mean())
        per_class[cls] = {
            "auc": auc(sign * mt, sign * nt), "advantage": tpr - fpr,
            "tpr": tpr, "fpr": fpr, "calibrated_sign": sign,
            "calibrated_threshold": float(threshold) if np.isfinite(threshold) else str(threshold),
            "calibration_count_per_group": len(mc), "test_count_per_group": len(mt),
        }
    groups = {}
    for group in ("target", "retained"):
        rows = [row for cls, row in per_class.items()
                if (int(cls) in target_classes) == (group == "target")]
        groups[group] = {
            "macro_auc": float(np.mean([r["auc"] for r in rows])) if rows else None,
            "macro_advantage": float(np.mean([r["advantage"] for r in rows])) if rows else None,
            "class_count": len(rows),
        }
    return {"attack": "class-conditioned calibrated negative cross entropy",
            "access": "black-box posterior and true class label",
            "per_class": per_class, "groups": groups}
