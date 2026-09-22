"""
models.py

Hyperparameter tuning and final fitting for the three models compared in the
paper:

    tune_classical(split)                    SVC with rbf / poly / linear kernels
    tune_quantum(split)                      SVC on a precomputed ZZFeatureMap fidelity kernel
    tune_dual(split, classical, quantum)     SVC on alpha * K_quantum + (1 - alpha) * K_classical

Selection uses stratified k-fold CV on the VRE0 training set only. All models
share the same fold splits for a given seed, so their CV scores are paired.
Test sets (VRE0 internal, VRE30 and VRE80 held-out) are only used after the
final model is chosen.

Each tune_* function returns a result dict with the same keys, which
stats.py and reporting.py consume.
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.svm import SVC

import config
from kernels import classical_kernel_from_params, combine_kernels, precompute_quantum_kernels


# (split key suffix, human-readable name) for the three test sets.
TEST_SETS = [
    ("0", "VRE0 internal test"),
    ("30", "VRE30 held-out test"),
    ("80", "VRE80 held-out test"),
]


# =========================
# Cross-validation helpers
# =========================

def make_cv(seed, n_splits=None):
    """Stratified folds. Using the split seed keeps folds identical across models."""
    return StratifiedKFold(n_splits=n_splits or config.CV_FOLDS, shuffle=True, random_state=seed)


def cv_scores_svc(X, y, params, seed, n_splits=None):
    """Fold accuracies for an SVC built from params (feature-space kernel)."""
    y = np.asarray(y)
    scores = []
    for train_idx, val_idx in make_cv(seed, n_splits).split(X, y):
        model = SVC(**params)
        model.fit(X[train_idx], y[train_idx])
        scores.append(accuracy_score(y[val_idx], model.predict(X[val_idx])))
    return np.asarray(scores, dtype=float)


def cv_scores_precomputed_svc(K_train, y, C, seed, n_splits=None):
    """Fold accuracies for an SVC on a precomputed (n_train x n_train) Gram matrix."""
    y = np.asarray(y)
    scores = []
    for train_idx, val_idx in make_cv(seed, n_splits).split(K_train, y):
        model = SVC(kernel="precomputed", C=C)
        model.fit(K_train[np.ix_(train_idx, train_idx)], y[train_idx])
        pred = model.predict(K_train[np.ix_(val_idx, train_idx)])
        scores.append(accuracy_score(y[val_idx], pred))
    return np.asarray(scores, dtype=float)


def _std(scores):
    return float(scores.std(ddof=1)) if len(scores) > 1 else 0.0


# =========================
# Scoring helpers
# =========================

def predict_and_score(model, X_or_K):
    """Predicted labels and decision scores (None if unavailable)."""
    y_pred = model.predict(X_or_K)
    try:
        y_score = model.decision_function(X_or_K)
    except Exception:
        y_score = None
    return y_pred, y_score


def safe_multiclass_roc_auc(y_true, y_score, score_classes, labels):
    """
    Macro one-vs-rest ROC AUC from SVC decision scores.

    sklearn's roc_auc_score(multi_class="ovr") only accepts probabilities that
    sum to 1, which SVC decision_function scores are not; the original script
    therefore always returned NaN. Here each class's AUC is computed as a
    binary problem (class vs rest) using that class's decision score, which is
    valid for any ranking score, and the per-class AUCs are averaged.

    Classes absent from y_true are skipped. Returns NaN if no class can be scored.
    """
    if y_score is None:
        return np.nan

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    score_classes = list(score_classes)

    if y_score.ndim == 1:
        if len(score_classes) != 2:
            return np.nan
        y_score = np.column_stack([-y_score, y_score])

    if y_score.shape[1] != len(score_classes):
        return np.nan

    aucs = []
    for label in labels:
        if label not in score_classes:
            continue
        is_label = y_true == label
        if is_label.all() or not is_label.any():
            continue  # AUC undefined when the class is absent or the only one present
        aucs.append(roc_auc_score(is_label, y_score[:, score_classes.index(label)]))

    return float(np.mean(aucs)) if aucs else np.nan


def build_result(name, model, split, params, best_row, test_inputs, t0, rows, **extra):
    """
    Evaluate a fitted model on all test sets and package the standard result dict.

    test_inputs maps suffix ("0", "30", "80") to the matrix passed to predict:
    features for feature-space SVCs, (n_test x n_train) Gram matrices otherwise.
    """
    labels = sorted(np.unique(split["y_train"]))
    result = {
        "name": name,
        "model": model,
        "classes": model.classes_,
        "params": params,
        "cv_mean": best_row["mean_cv"],
        "cv_std": best_row["std_cv"],
        "cv_folds": best_row["fold_scores"],
    }
    for suffix, _ in TEST_SETS:
        y_true = split["y_test" + suffix]
        y_pred, y_score = predict_and_score(model, test_inputs[suffix])
        result["y_pred" + suffix] = y_pred
        result["y_score" + suffix] = y_score
        result["acc" + suffix] = accuracy_score(y_true, y_pred)
        result["auc" + suffix] = safe_multiclass_roc_auc(y_true, y_score, model.classes_, labels)

    result["elapsed_sec"] = time.time() - t0
    result["results_df"] = pd.DataFrame(rows).sort_values("mean_cv", ascending=False).reset_index(drop=True)
    result.update(extra)
    return result


def _is_better(row, best):
    # Strict ">" keeps the first grid point on ties, matching the original script.
    return best is None or row["mean_cv"] > best["mean_cv"]


# =========================
# Classical SVC
# =========================

def tune_classical(split, param_grids=None):
    """Grid search over rbf / poly / linear SVCs with k-fold CV."""
    t0 = time.time()
    param_grids = param_grids if param_grids is not None else config.CLASSICAL_PARAM_GRIDS
    X_train, y_train, seed = split["X_train"], split["y_train"], split["seed"]

    rows, best = [], None
    for grid in param_grids:
        for params in ParameterGrid(grid):
            scores = cv_scores_svc(X_train, y_train, params, seed)
            row = {"params": params, "mean_cv": float(scores.mean()), "std_cv": _std(scores), "fold_scores": scores}
            rows.append(row)
            if _is_better(row, best):
                best = row

    model = SVC(**best["params"])
    model.fit(X_train, y_train)

    return build_result(
        name="Classical SVC ({})".format(best["params"]["kernel"]),
        model=model,
        split=split,
        params=best["params"],
        best_row=best,
        test_inputs={s: split["X_test" + s] for s, _ in TEST_SETS},
        t0=t0,
        rows=rows,
    )


# =========================
# Quantum SVC
# =========================

def tune_quantum(split, mode=None, entanglements=None, C_grid=None, reps=None):
    """
    Precompute one fidelity kernel per entanglement pattern, then tune C by CV.
    Kernel matrices are cached on disk (see kernels.py) and returned in
    result["cache"] so tune_dual can reuse them without recomputation.
    """
    t0 = time.time()
    mode = (mode or config.QUANTUM_KERNEL_MODE).lower()
    entanglements = entanglements or config.QUANTUM_ENTANGLEMENTS
    C_grid = C_grid or config.QUANTUM_C_GRID
    reps = reps or config.QUANTUM_FEATUREMAP_REPS
    y_train, seed = split["y_train"], split["seed"]

    rows, best, cache = [], None, {}
    for ent in entanglements:
        print("  precomputing quantum kernel: mode={}, entanglement={}".format(mode, ent))
        K_train, K_test0, K_test30, K_test80, qkernel = precompute_quantum_kernels(
            split["X_train"], split["X_test0"], split["X_test30"], split["X_test80"],
            entanglement=ent, reps=reps, mode=mode,
        )
        cache[ent] = {"K_train": K_train, "K_test0": K_test0, "K_test30": K_test30,
                      "K_test80": K_test80, "kernel": qkernel}

        for C in C_grid:
            scores = cv_scores_precomputed_svc(K_train, y_train, C, seed)
            params = {"C": C, "entanglement": ent, "reps": reps, "quantum_kernel_mode": mode}
            row = {"params": params, "C": C, "entanglement": ent, "quantum_kernel_mode": mode,
                   "mean_cv": float(scores.mean()), "std_cv": _std(scores), "fold_scores": scores}
            rows.append(row)
            if _is_better(row, best):
                best = row

    ent = best["entanglement"]
    model = SVC(kernel="precomputed", C=best["C"])
    model.fit(cache[ent]["K_train"], y_train)

    return build_result(
        name="Quantum SVC ({}, {})".format(mode, ent),
        model=model,
        split=split,
        params=best["params"],
        best_row=best,
        test_inputs={s: cache[ent]["K_test" + s] for s, _ in TEST_SETS},
        t0=t0,
        rows=rows,
        cache=cache,
        best_entanglement=ent,
    )


# =========================
# Dual kernel SVC
# =========================

def tune_dual(split, classical_result, quantum_result, C_grid=None, alpha_grid=None):
    """
    Tune C and alpha for K = alpha * K_quantum + (1 - alpha) * K_classical.

    The quantum half uses the entanglement chosen by tune_quantum and the
    classical half uses the kernel hyperparameters chosen by tune_classical,
    so only C and alpha are searched here. alpha = 0 reproduces the classical
    kernel and alpha = 1 the quantum kernel, so both baselines lie on the grid.
    """
    t0 = time.time()
    C_grid = C_grid or config.DUAL_C_GRID
    alpha_grid = alpha_grid if alpha_grid is not None else config.DUAL_ALPHA_GRID
    X_train, y_train, seed = split["X_train"], split["y_train"], split["seed"]

    ent = quantum_result["best_entanglement"]
    Kq = {"train": quantum_result["cache"][ent]["K_train"]}
    Kq.update({s: quantum_result["cache"][ent]["K_test" + s] for s, _ in TEST_SETS})

    classical_params = classical_result["params"]
    Kc = {"train": classical_kernel_from_params(X_train, X_train, classical_params)}
    Kc.update({s: classical_kernel_from_params(split["X_test" + s], X_train, classical_params) for s, _ in TEST_SETS})

    mode = quantum_result["params"].get("quantum_kernel_mode", config.QUANTUM_KERNEL_MODE)
    rows, best = [], None

    for C in C_grid:
        for alpha in alpha_grid:
            alpha = float(alpha)
            K_train = combine_kernels(Kq["train"], Kc["train"], alpha)
            scores = cv_scores_precomputed_svc(K_train, y_train, C, seed)

            # Diagnostic only, for the alpha-curve figure. Never used for selection.
            diag_model = SVC(kernel="precomputed", C=C).fit(K_train, y_train)
            test0_acc = accuracy_score(split["y_test0"], diag_model.predict(combine_kernels(Kq["0"], Kc["0"], alpha)))

            params = {"C": C, "alpha": alpha, "entanglement": ent,
                      "classical_kernel": classical_params.get("kernel"), "quantum_kernel_mode": mode}
            row = {"params": params, "C": C, "alpha": alpha, "quantum_kernel_mode": mode,
                   "mean_cv": float(scores.mean()), "std_cv": _std(scores), "fold_scores": scores,
                   "test0_acc": test0_acc}
            rows.append(row)
            if _is_better(row, best):
                best = row

    alpha = best["alpha"]
    model = SVC(kernel="precomputed", C=best["C"])
    model.fit(combine_kernels(Kq["train"], Kc["train"], alpha), y_train)

    return build_result(
        name="Dual Kernel SVC ({}, alpha={:.2f})".format(mode, alpha),
        model=model,
        split=split,
        params=best["params"],
        best_row=best,
        test_inputs={s: combine_kernels(Kq[s], Kc[s], alpha) for s, _ in TEST_SETS},
        t0=t0,
        rows=rows,
    )


# =========================
# Convenience
# =========================

def tune_all(split, mode=None, verbose=True):
    """Run classical, quantum and dual tuning for one split."""
    if verbose:
        print("Tuning classical SVC...")
    classical = tune_classical(split)
    if verbose:
        print("Tuning quantum SVC...")
    quantum = tune_quantum(split, mode=mode)
    if verbose:
        print("Tuning dual kernel SVC...")
    dual = tune_dual(split, classical, quantum)
    return classical, quantum, dual


def summarize_results(*results):
    """One-row-per-model comparison table of CV and test metrics."""
    rows = []
    for r in results:
        row = {"Model": r["name"], "CV Accuracy": r["cv_mean"], "CV Std": r["cv_std"]}
        for suffix, _ in TEST_SETS:
            row["VRE{} Accuracy".format(suffix)] = r["acc" + suffix]
        for suffix, _ in TEST_SETS:
            row["VRE{} ROC AUC".format(suffix)] = r["auc" + suffix]
        row["Tuning Seconds"] = r["elapsed_sec"]
        rows.append(row)
    return pd.DataFrame(rows)
