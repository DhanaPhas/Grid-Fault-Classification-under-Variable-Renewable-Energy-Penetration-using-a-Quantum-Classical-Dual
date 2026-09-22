"""
stats.py

Statistical comparison of the classical, quantum and dual-kernel models.

Two levels of inference are provided:

1. Across seeds (primary). Each seed gives a fresh train/test draw, so the
   per-seed metric differences between two models are paired observations.
   repeated_seed_pairwise_metric_tests() applies an exact sign-flip
   permutation test to those differences with Holm correction across the
   three model pairs. This is the test to base claims on.

2. Within one fixed test set (secondary). final_fixed_test_pairwise_table()
   uses McNemar's exact test on paired correctness and paired bootstrap CIs.
   These describe one split only and ignore split-to-split variability.

Power note: with n seeds the smallest achievable two-sided sign-flip p-value
is 2 / 2**n. With Holm over m pairs the smallest adjusted p-value is
m * 2 / 2**n, so 6 seeds and 3 pairs can never reach p < 0.05 (min 0.094).
Use min_achievable_holm_p() to check before running; >= 7 seeds are needed
for 3 pairs, and 10+ is advisable.
"""

import itertools
import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import config
from models import TEST_SETS, safe_multiclass_roc_auc


MODEL_KEYS = [("Classical", "classical"), ("Quantum", "quantum"), ("Dual", "dual")]

MODEL_PAIRS = [
    ("Dual - Classical", "dual", "classical"),
    ("Dual - Quantum", "dual", "quantum"),
    ("Quantum - Classical", "quantum", "classical"),
]

# (display name, column suffix in the per-seed summary table)
SEED_METRICS = [
    ("CV Accuracy", "cv"),
    ("VRE0 Test Accuracy", "acc0"),
    ("VRE30 Held-out Accuracy", "acc30"),
    ("VRE80 Held-out Accuracy", "acc80"),
    ("VRE0 ROC AUC", "auc0"),
    ("VRE30 ROC AUC", "auc30"),
    ("VRE80 ROC AUC", "auc80"),
]


def _percentile_ci(samples, ci):
    alpha = 1.0 - ci
    return float(np.percentile(samples, 100 * alpha / 2)), float(np.percentile(samples, 100 * (1 - alpha / 2)))


def _labels(split):
    return sorted(np.unique(split["y_train"]))


# =========================
# Power check
# =========================

def min_achievable_p(n_seeds):
    """Smallest two-sided exact sign-flip p-value possible with n paired observations."""
    return min(1.0, 2.0 / (2 ** n_seeds)) if n_seeds > 0 else np.nan


def min_achievable_holm_p(n_seeds, n_comparisons=len(MODEL_PAIRS)):
    return min(1.0, n_comparisons * min_achievable_p(n_seeds))


# =========================
# Single test set: bootstrap
# =========================

def bootstrap_accuracy_ci(y_true, y_pred, n_bootstrap=None, ci=None, random_state=42):
    """Percentile bootstrap CI for accuracy on one test set."""
    n_bootstrap = n_bootstrap or config.BOOTSTRAP_N
    ci = ci or config.BOOTSTRAP_CI
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, len(correct), size=(n_bootstrap, len(correct)))
    boot = correct[idx].mean(axis=1)
    lower, upper = _percentile_ci(boot, ci)
    return {
        "test_accuracy": float(correct.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_width": upper - lower,
        "bootstrap_mean": float(boot.mean()),
        "bootstrap_std": float(boot.std(ddof=1)),
    }


def _metric(y_true, y_pred, y_score, classes, labels, metric):
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "roc_auc":
        return safe_multiclass_roc_auc(y_true, y_score, classes, labels)
    raise ValueError("Unknown metric: {}".format(metric))


def paired_bootstrap_delta_ci(y_true, result_a, result_b, suffix, metric, labels,
                              n_bootstrap=None, ci=None, random_state=42):
    """
    Paired bootstrap CI for metric(A) - metric(B) on one test set. The same
    resampled indices are used for both models, preserving the pairing.
    """
    n_bootstrap = n_bootstrap or config.BOOTSTRAP_N
    ci = ci or config.BOOTSTRAP_CI
    y_true = np.asarray(y_true)

    def arrays(result):
        pred = np.asarray(result["y_pred" + suffix])
        score = result.get("y_score" + suffix)
        return pred, (None if score is None else np.asarray(score)), result["classes"]

    pred_a, score_a, cls_a = arrays(result_a)
    pred_b, score_b, cls_b = arrays(result_b)

    observed = (_metric(y_true, pred_a, score_a, cls_a, labels, metric)
                - _metric(y_true, pred_b, score_b, cls_b, labels, metric))

    if metric == "accuracy":
        # Vectorized fast path.
        diff = (pred_a == y_true).astype(float) - (pred_b == y_true).astype(float)
        rng = np.random.default_rng(random_state)
        idx = rng.integers(0, len(diff), size=(n_bootstrap, len(diff)))
        deltas = diff[idx].mean(axis=1)
    else:
        rng = np.random.default_rng(random_state)
        deltas = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(y_true), size=len(y_true))
            d = (_metric(y_true[idx], pred_a[idx], None if score_a is None else score_a[idx], cls_a, labels, metric)
                 - _metric(y_true[idx], pred_b[idx], None if score_b is None else score_b[idx], cls_b, labels, metric))
            if np.isfinite(d):
                deltas.append(d)
        deltas = np.asarray(deltas, dtype=float)

    if len(deltas) == 0:
        return {"delta": observed, "ci_lower": np.nan, "ci_upper": np.nan,
                "bootstrap_mean": np.nan, "bootstrap_std": np.nan}
    lower, upper = _percentile_ci(deltas, ci)
    return {
        "delta": float(observed),
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_mean": float(deltas.mean()),
        "bootstrap_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else np.nan,
    }


def exact_mcnemar_p_value(y_true, y_pred_a, y_pred_b):
    """
    Two-sided exact McNemar test on paired correctness.
    Returns (p_value, n_only_a_correct, n_only_b_correct).
    """
    y_true = np.asarray(y_true)
    a_ok = np.asarray(y_pred_a) == y_true
    b_ok = np.asarray(y_pred_b) == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    if n == 0:
        return 1.0, b, c
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail), b, c


# =========================
# Multiple-comparison correction
# =========================

def holm_bonferroni(p_values):
    """Holm step-down adjusted p-values. NaNs are ignored and stay NaN."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid = np.where(np.isfinite(p_values))[0]
    m = len(valid)
    running_max = 0.0
    for rank, idx in enumerate(valid[np.argsort(p_values[valid])]):
        running_max = max(running_max, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def add_holm_correction(df, p_col, group_cols, out_col):
    df = df.copy()
    df[out_col] = np.nan
    if df.empty or p_col not in df.columns:
        return df
    for _, idx in df.groupby(group_cols).groups.items():
        locs = list(idx)
        df.loc[locs, out_col] = holm_bonferroni(df.loc[locs, p_col].astype(float).values)
    return df


# =========================
# Single seed tables
# =========================

def model_ci_table(split, results):
    """Per model and test set: accuracy, ROC AUC and bootstrap accuracy CI."""
    labels = _labels(split)
    rows = []
    for suffix, dataset_name in TEST_SETS:
        y_true = split["y_test" + suffix]
        for _, key in MODEL_KEYS:
            r = results[key]
            s = bootstrap_accuracy_ci(y_true, r["y_pred" + suffix], random_state=split["seed"])
            rows.append({
                "Dataset": dataset_name,
                "Model": r["name"],
                "Test Accuracy": s["test_accuracy"],
                "ROC AUC OVR Macro": safe_multiclass_roc_auc(y_true, r.get("y_score" + suffix), r["classes"], labels),
                "95% CI Lower": s["ci_lower"],
                "95% CI Upper": s["ci_upper"],
                "CI Width": s["ci_width"],
                "Bootstrap Mean": s["bootstrap_mean"],
                "Bootstrap Std": s["bootstrap_std"],
            })
    return pd.DataFrame(rows)


def dual_vs_classical_delta_table(split, results):
    """Paired bootstrap CI of dual minus classical accuracy on each test set."""
    labels = _labels(split)
    rows = []
    for suffix, dataset_name in TEST_SETS:
        s = paired_bootstrap_delta_ci(split["y_test" + suffix], results["dual"], results["classical"],
                                      suffix, "accuracy", labels, random_state=split["seed"])
        rows.append({
            "Dataset": dataset_name,
            "Dual - Classical Accuracy": s["delta"],
            "95% Delta CI Lower": s["ci_lower"],
            "95% Delta CI Upper": s["ci_upper"],
            "Strict 95% CI Win": bool(s["ci_lower"] > 0),
        })
    return pd.DataFrame(rows)


def strict_win_flags(results, delta_df):
    """
    Dual beats classical on this seed if CV folds are all >= (one strictly >)
    and the paired bootstrap CI lower bound is > 0 on every test set.
    """
    d, c = results["dual"], results["classical"]
    flags = {
        "strict_cv_win": bool(
            d["cv_mean"] > c["cv_mean"]
            and np.all(d["cv_folds"] >= c["cv_folds"])
            and np.any(d["cv_folds"] > c["cv_folds"])
        )
    }
    for suffix, dataset_name in TEST_SETS:
        flags["strict_test{}_95ci_win".format(suffix)] = bool(
            delta_df.loc[delta_df["Dataset"] == dataset_name, "Strict 95% CI Win"].iloc[0]
        )
    flags["strict_all_win"] = all(flags.values())
    return flags


def final_fixed_test_pairwise_table(split, results, random_state=42):
    """All model pairs on the fixed test sets: McNemar + paired bootstrap CIs, Holm per dataset/metric."""
    labels = _labels(split)
    rows = []
    for d_idx, (suffix, dataset_name) in enumerate(TEST_SETS):
        y_true = split["y_test" + suffix]
        for p_idx, (pair_name, a, b) in enumerate(MODEL_PAIRS):
            ra, rb = results[a], results[b]
            p, a_only, b_only = exact_mcnemar_p_value(y_true, ra["y_pred" + suffix], rb["y_pred" + suffix])
            acc = paired_bootstrap_delta_ci(y_true, ra, rb, suffix, "accuracy", labels,
                                            random_state=random_state + 100 * d_idx + p_idx)
            auc = paired_bootstrap_delta_ci(y_true, ra, rb, suffix, "roc_auc", labels,
                                            random_state=random_state + 1000 + 100 * d_idx + p_idx)
            rows.append({
                "Dataset": dataset_name, "Metric": "Accuracy", "Pair": pair_name,
                "Difference": acc["delta"], "95% Delta CI Lower": acc["ci_lower"], "95% Delta CI Upper": acc["ci_upper"],
                "McNemar p-value": p, "Discordant A Correct Only": a_only, "Discordant B Correct Only": b_only,
            })
            rows.append({
                "Dataset": dataset_name, "Metric": "ROC AUC", "Pair": pair_name,
                "Difference": auc["delta"], "95% Delta CI Lower": auc["ci_lower"], "95% Delta CI Upper": auc["ci_upper"],
                "McNemar p-value": np.nan, "Discordant A Correct Only": np.nan, "Discordant B Correct Only": np.nan,
            })
    return add_holm_correction(pd.DataFrame(rows), "McNemar p-value", ["Dataset", "Metric"], "Holm McNemar p-value")


# =========================
# Across seeds (primary inference)
# =========================

def bootstrap_mean_ci(values, n_bootstrap=None, ci=None, random_state=42):
    """
    Bootstrap CI for a mean. With few seeds (e.g. 6) this interval is too
    narrow in practice; report it alongside the permutation p-value.
    """
    n_bootstrap = n_bootstrap or config.BOOTSTRAP_N
    ci = ci or config.BOOTSTRAP_CI
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    rng = np.random.default_rng(random_state)
    boot = values[rng.integers(0, len(values), size=(n_bootstrap, len(values)))].mean(axis=1)
    lower, upper = _percentile_ci(boot, ci)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "ci_lower": lower, "ci_upper": upper}


def paired_sign_flip_p_value(deltas, random_state=42, max_exact=20, n_permutations=100000):
    """
    Two-sided paired sign-flip permutation test of mean(deltas) == 0.
    Exact enumeration for n <= max_exact, Monte Carlo otherwise.
    """
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    n = len(deltas)
    if n == 0:
        return np.nan
    observed = abs(deltas.mean())

    if n <= max_exact:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        permuted = np.abs(signs @ deltas / n)
        return float(np.mean(permuted >= observed - 1e-12))

    rng = np.random.default_rng(random_state)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    permuted = np.abs(signs @ deltas / n)
    return float((np.sum(permuted >= observed - 1e-12) + 1) / (n_permutations + 1))


def paired_t_test_p_value(deltas):
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) < 2 or np.allclose(deltas, deltas[0]):
        return np.nan
    from scipy import stats as sps
    return float(sps.ttest_1samp(deltas, popmean=0.0).pvalue)


def wilcoxon_p_value(deltas):
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) < 2 or np.allclose(deltas, 0.0):
        return np.nan
    from scipy import stats as sps
    try:
        return float(sps.wilcoxon(deltas).pvalue)
    except ValueError:
        return np.nan


def repeated_seed_model_summary(seed_df):
    """Mean, std, min and max of each metric across seeds, per model."""
    rows = []
    for metric_name, suffix in SEED_METRICS:
        for model_name, prefix in MODEL_KEYS:
            col = "{}_{}".format(prefix, suffix)
            if col not in seed_df.columns:
                continue
            v = seed_df[col].astype(float).values
            rows.append({
                "Metric": metric_name, "Model": model_name, "N Seeds": int(np.isfinite(v).sum()),
                "Mean": float(np.nanmean(v)),
                "Std Across Seeds": float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else 0.0,
                "Min": float(np.nanmin(v)), "Max": float(np.nanmax(v)),
            })
    return pd.DataFrame(rows)


def repeated_seed_pairwise_metric_tests(seed_df, random_state=42):
    """
    Primary inference table: for every metric and model pair, the mean paired
    difference across seeds, bootstrap CI, exact sign-flip p-value (Holm
    corrected across the pairs within each metric), plus t-test and Wilcoxon
    p-values for reference.
    """
    rows = []
    for m_idx, (metric_name, suffix) in enumerate(SEED_METRICS):
        for p_idx, (pair_name, a, b) in enumerate(MODEL_PAIRS):
            a_col, b_col = "{}_{}".format(a, suffix), "{}_{}".format(b, suffix)
            if a_col not in seed_df.columns or b_col not in seed_df.columns:
                continue
            deltas = seed_df[a_col].astype(float).values - seed_df[b_col].astype(float).values
            deltas = deltas[np.isfinite(deltas)]
            rs = random_state + 100 * m_idx + p_idx
            s = bootstrap_mean_ci(deltas, random_state=rs)
            rows.append({
                "Metric": metric_name, "Pair": pair_name, "N Seeds": len(deltas),
                "Mean Difference": s["mean"], "Std Difference": s["std"],
                "95% Mean Delta CI Lower": s["ci_lower"], "95% Mean Delta CI Upper": s["ci_upper"],
                "Paired Permutation p-value": paired_sign_flip_p_value(deltas, random_state=rs),
                "Paired t-test p-value": paired_t_test_p_value(deltas),
                "Wilcoxon p-value": wilcoxon_p_value(deltas),
            })
    out = add_holm_correction(pd.DataFrame(rows), "Paired Permutation p-value", ["Metric"], "Holm p-value")
    if not out.empty:
        out["Supported Difference"] = (
            (out["Holm p-value"] < config.SIGNIFICANCE_LEVEL)
            & ((out["95% Mean Delta CI Lower"] > 0) | (out["95% Mean Delta CI Upper"] < 0))
        )
    return out


def repeated_seed_dual_delta_tests(seed_df, random_state=42):
    """Dual minus classical only, accuracy metrics (subset of the pairwise table, no Holm)."""
    rows = []
    for i, (metric_name, suffix) in enumerate(SEED_METRICS[:4]):
        deltas = seed_df["dual_" + suffix].astype(float).values - seed_df["classical_" + suffix].astype(float).values
        s = bootstrap_mean_ci(deltas, random_state=random_state + i)
        p = paired_sign_flip_p_value(deltas, random_state=random_state + i)
        rows.append({
            "Metric": metric_name, "N Seeds": len(deltas),
            "Mean Dual - Classical": s["mean"], "Std Delta Across Seeds": s["std"],
            "95% Mean Delta CI Lower": s["ci_lower"], "95% Mean Delta CI Upper": s["ci_upper"],
            "Paired Sign-flip p-value": p,
            "Significant Positive Delta": bool(s["ci_lower"] > 0 and p < config.SIGNIFICANCE_LEVEL),
        })
    return pd.DataFrame(rows)


def model_cost_summary(seed_df):
    """
    Tuning time per seed. dual_elapsed_sec only covers the alpha/C search,
    because it reuses the classical hyperparameters and the cached quantum
    kernels, so an extra "Dual (end-to-end)" row adds all three stages.
    That row is the fair cost to report for the dual model.
    """
    series = {name: seed_df[prefix + "_elapsed_sec"].astype(float).values
              for name, prefix in MODEL_KEYS if prefix + "_elapsed_sec" in seed_df.columns}
    if len(series) == 3:
        series["Dual (end-to-end)"] = series["Classical"] + series["Quantum"] + series["Dual"]

    rows = []
    for model_name, v in series.items():
        rows.append({
            "Model": model_name, "Mean Seconds per Seed": float(v.mean()),
            "Std Seconds": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "Min Seconds": float(v.min()), "Max Seconds": float(v.max()),
        })
    return pd.DataFrame(rows)
