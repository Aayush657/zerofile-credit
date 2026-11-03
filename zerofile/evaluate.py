"""Metrics, policy curves and the fairness audit. Used at training time only."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def bootstrap_auc(y, score, resamples: int = 300, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    draws = [roc_auc_score(y[i], score[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(resamples))]
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"value": float(roc_auc_score(y, score)), "low": float(low), "high": float(high)}


def bootstrap_auc_gain(y, score_a, score_b, resamples: int = 300, seed: int = 7) -> dict:
    """Paired: the same resample scores both models, so the CI is on the gap itself."""
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(resamples):
        i = rng.integers(0, len(y), len(y))
        gaps.append(roc_auc_score(y[i], score_a[i]) - roc_auc_score(y[i], score_b[i]))
    low, high = np.percentile(gaps, [2.5, 97.5])
    return {
        "value": float(roc_auc_score(y, score_a) - roc_auc_score(y, score_b)),
        "low": float(low),
        "high": float(high),
    }


def ks_statistic(y, score) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.max(tpr - fpr))


def brier(y, p) -> float:
    return float(np.mean((p - y) ** 2))


def calibration_table(y, p, bins: int = 10) -> list[dict]:
    """Equal-sized bins of predicted risk."""
    groups = pd.qcut(p, bins, labels=False, duplicates="drop")
    table = (
        pd.DataFrame({"group": groups, "p": p, "y": y})
        .groupby("group")
        .agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    )
    return [{"predicted": float(r.predicted), "actual": float(r.actual), "n": int(r.n)} for r in table.itertuples()]


def expected_calibration_error(y, p, bins: int = 10) -> float:
    table = calibration_table(y, p, bins)
    total = sum(row["n"] for row in table)
    return float(sum(row["n"] * abs(row["predicted"] - row["actual"]) for row in table) / total)


def pick_threshold(p, y, target: float, min_rows: int = 500) -> float:
    """Highest risk cutoff at which the group approved so far still defaults at or below target."""
    order = np.argsort(p)
    rate = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    passing = np.flatnonzero(rate[min_rows:] <= target)
    if passing.size == 0:
        raise ValueError(f"No cutoff reaches a default rate of {target:.3f}")
    return float(p[order][passing[-1] + min_rows])


def policy_row(approved, y, loan) -> dict:
    count = int(approved.sum())
    return {
        "share": float(approved.mean()),
        "people": count,
        "default": float(y[approved].mean()) if count else None,
        "crore": float(loan[approved].sum() / 1e7),
    }


def policy_curve(p, y, loan, thresholds) -> list[dict]:
    return [{"threshold": float(t), **policy_row(p <= t, y, loan)} for t in thresholds]


def rule_default_at(rule_value, y, share: float) -> float:
    """Default rate if a simple rule approved the same share, lowest values first."""
    return float(y[np.argsort(rule_value)[: max(1, round(share * len(y)))]].mean())


def classification_at(approved, y) -> dict:
    """Treat 'not approved' as the prediction of default."""
    flagged = ~approved
    caught = y[flagged].sum()
    precision = caught / max(1, flagged.sum())
    recall = caught / max(1, y.sum())
    return {
        "accuracy": float(np.mean(flagged == y.astype(bool))),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(1e-9, precision + recall)),
    }


def group_audit(groups, approved, y, min_rows: int = 200) -> list[dict]:
    """Approval, default and four-fifths check per group. Small groups are skipped because their rates are noise."""
    frame = pd.DataFrame({"group": groups, "approved": approved, "default": y})
    rows = []
    for name, part in frame.groupby("group", observed=True):
        if len(part) < min_rows:
            continue
        repaid = part[part["default"] == 0]
        rows.append(
            {
                "group": str(name),
                "n": int(len(part)),
                "approval": float(part["approved"].mean()),
                "default": float(part["default"].mean()),
                "default_if_approved": float(part.loc[part["approved"], "default"].mean())
                if part["approved"].any()
                else None,
                "approval_if_repaid": float(repaid["approved"].mean()),
            }
        )
    top = max(row["approval"] for row in rows)
    for row in rows:
        row["ratio"] = row["approval"] / top
        row["passes_four_fifths"] = row["ratio"] >= 0.8
    return rows


def regional_floor(p, y, loan, regions, approved, floor_ratio: float = 0.8, min_rows: int = 200) -> dict:
    """Price of a fairness fix: every region must approve at least floor_ratio of the overall rate, safest applicants first."""
    floor = floor_ratio * float(approved.mean())
    result = approved.copy()
    below_before, people_added = 0, 0
    for region in np.unique(regions):
        members = np.flatnonzero(regions == region)
        if len(members) < min_rows:
            continue
        needed = int(np.ceil(floor * len(members))) - int(result[members].sum())
        if needed > 0:
            below_before += 1
            people_added += needed
            waiting = members[~result[members]]
            result[waiting[np.argsort(p[waiting])[:needed]]] = True
    audit = group_audit(regions, result, y, min_rows)
    return {
        "floor": floor,
        "regions_below_before": below_before,
        "regions_below_after": sum(row["approval"] < floor for row in audit),
        "people_added": people_added,
        "worst_ratio": min(row["ratio"] for row in audit),
        **policy_row(result, y, loan),
    }
