"""
reporting.py

Display and save tables and figures. Nothing here computes statistics; it
only formats what models.py and stats.py produce.

Every function works both in Jupyter (styled tables, inline figures) and in a
plain terminal (printed tables, figures saved to disk).
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import config
from models import TEST_SETS
from stats import MODEL_KEYS

try:
    from IPython import get_ipython
    from IPython.display import display as _ipy_display

    IN_NOTEBOOK = get_ipython() is not None and "IPKernelApp" in get_ipython().config
except Exception:
    IN_NOTEBOOK = False
    _ipy_display = None


# =========================
# Tables
# =========================

def show_table(df, title=None, float_format="{:.4f}"):
    """Styled table in Jupyter, plain text in a terminal."""
    if title:
        print("\n" + "=" * 16 + " {} ".format(title) + "=" * 16)
    if df is None or df.empty:
        print("(empty)")
        return
    float_cols = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    if IN_NOTEBOOK:
        _ipy_display(df.style.format({c: float_format for c in float_cols}, na_rep="NaN"))
    else:
        with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 60):
            print(df.to_string(index=False, float_format=lambda v: float_format.format(v)))


def save_table(df, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name if name.endswith(".csv") else name + ".csv")
    df.to_csv(path, index=False)
    return path


def save_json(obj, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


def seed_summary_row(split, results, flags):
    """Flat one-row summary for a seed (the per-seed CSV row)."""
    row = {
        "dataset": config.SELECTED_DATASET,
        "quantum_kernel": config.QUANTUM_KERNEL_MODE,
        "seed": split["seed"],
        "pca_features": split["pca_features"],
    }
    for _, key in MODEL_KEYS:
        r = results[key]
        row[key + "_name"] = r["name"]
        row[key + "_params"] = json.dumps(r["params"], default=str, sort_keys=True)
        row[key + "_cv"] = r["cv_mean"]
        row[key + "_cv_std"] = r["cv_std"]
        for suffix, _ in TEST_SETS:
            row["{}_acc{}".format(key, suffix)] = r["acc" + suffix]
            row["{}_auc{}".format(key, suffix)] = r["auc" + suffix]
        row[key + "_elapsed_sec"] = r["elapsed_sec"]
    row.update(flags)
    return row


def prediction_table_for_seed(split, results):
    """Long table of every test prediction, for later re-analysis."""
    frames = []
    for suffix, dataset_name in TEST_SETS:
        y_true = np.asarray(split["y_test" + suffix])
        for model_name, key in MODEL_KEYS:
            r = results[key]
            y_pred = np.asarray(r["y_pred" + suffix])
            scores = r.get("y_score" + suffix)
            if scores is not None and np.ndim(scores) == 2:
                cls = [str(c) for c in r["classes"]]
                score_json = [json.dumps(dict(zip(cls, map(float, s))), sort_keys=True) for s in scores]
            else:
                score_json = [""] * len(y_true)
            frames.append(pd.DataFrame({
                "seed": split["seed"],
                "dataset": config.SELECTED_DATASET,
                "test_set": dataset_name,
                "model": model_name,
                "sample_index": np.arange(len(y_true)),
                "y_true": y_true,
                "y_pred": y_pred,
                "correct": y_true == y_pred,
                "score_json": score_json,
            }))
    return pd.concat(frames, ignore_index=True)


def model_comparison_table(results):
    rows = []
    for _, key in MODEL_KEYS:
        r = results[key]
        row = {"Model": r["name"], "CV Accuracy": r["cv_mean"], "CV Std": r["cv_std"]}
        for suffix, _ in TEST_SETS:
            row["VRE{} Test Accuracy".format(suffix)] = r["acc" + suffix]
        for suffix, _ in TEST_SETS:
            row["VRE{} ROC AUC".format(suffix)] = r["auc" + suffix]
        row["Tuning Seconds"] = r["elapsed_sec"]
        rows.append(row)
    return pd.DataFrame(rows)


def classification_reports(split, results, out_dir=None, verbose=True):
    """Per-class precision/recall/F1 for every model and test set."""
    labels = sorted(np.unique(split["y_train"]))
    text_blocks, rows = [], []
    for suffix, dataset_name in TEST_SETS:
        for model_name, key in MODEL_KEYS:
            y_true, y_pred = split["y_test" + suffix], results[key]["y_pred" + suffix]
            header = ">>> {}: {}".format(dataset_name, model_name)
            text = classification_report(y_true, y_pred, labels=labels, target_names=labels, zero_division=0)
            text_blocks.append(header + "\n" + text)
            rep = classification_report(y_true, y_pred, labels=labels, target_names=labels,
                                        zero_division=0, output_dict=True)
            for cls in labels:
                rows.append({"Dataset": dataset_name, "Model": model_name, "Class": cls,
                             "Precision": rep[cls]["precision"], "Recall": rep[cls]["recall"],
                             "F1": rep[cls]["f1-score"], "Support": int(rep[cls]["support"])})
    if verbose:
        print("\n\n".join(text_blocks))
    df = pd.DataFrame(rows)
    if out_dir:
        save_table(df, "reporting_seed_classification_report", out_dir)
        with open(os.path.join(out_dir, "reporting_seed_classification_report.txt"), "w") as fh:
            fh.write("\n\n".join(text_blocks))
    return df



def _finish_figure(fig, save_path, show):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# =========================
# Figures
# =========================

def plot_alpha_curve(split, dual_result, save_path=None, show=True):
    """
    CV accuracy vs alpha at the selected C. The VRE0 test curve is a
    diagnostic: alpha was selected by CV only.
    """
    df = dual_result["results_df"]
    curve = df[df["C"] == dual_result["params"]["C"]].sort_values("alpha")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["alpha"], curve["mean_cv"], "o-", label="VRE0 CV accuracy (used for selection)")
    ax.fill_between(curve["alpha"], curve["mean_cv"] - curve["std_cv"], curve["mean_cv"] + curve["std_cv"],
                    alpha=0.15)
    ax.plot(curve["alpha"], curve["test0_acc"], "s--", label="VRE0 test accuracy (diagnostic)")
    ax.axvline(dual_result["params"]["alpha"], color="red", linestyle=":", label="Selected alpha")
    ax.set_xlabel("alpha (quantum kernel weight)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Dual kernel accuracy vs alpha, {} seed {} (C={})".format(
        config.SELECTED_DATASET, split["seed"], dual_result["params"]["C"]))
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    _finish_figure(fig, save_path, show)


def plot_repeated_seed_summary(seed_df, metric="acc", save_path=None, show=True):
    """Mean +/- std across seeds for each model and test set."""
    groups = [("CV", "cv")]
    groups += [(name.split()[0], metric + suffix) for suffix, name in TEST_SETS]
    width = 0.8 / len(MODEL_KEYS)
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, (model_name, key) in enumerate(MODEL_KEYS):
        cols = ["{}_{}".format(key, g[1]) for g in groups]
        vals = seed_df[cols].astype(float)
        ax.bar(x + (k - 1) * width, vals.mean().values, width, yerr=vals.std(ddof=1).values,
               capsize=3, label=model_name)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Accuracy")
    ax.set_title("Mean +/- std over {} seeds, {}, {}".format(
        len(seed_df), config.SELECTED_DATASET, config.QUANTUM_KERNEL_MODE))
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    _finish_figure(fig, save_path, show)
