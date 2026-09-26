"""
run_experiment.py

Entry point for the full pipeline: for each seed, generate data, tune the
classical / quantum / dual-kernel SVCs, and evaluate them on the VRE0, VRE30
and VRE80 test sets; then run the across-seed statistics and write a
detailed report for one reporting seed.

Examples
--------
    # quick smoke test (tiny data, reduced grids; a few minutes)
    python run_experiment.py --dataset ieee38 --quick

    # full run as in the paper
    python run_experiment.py --dataset ieee38 --kernel FSK
    python run_experiment.py --dataset ieee68 --kernel FQK --seeds 14 22 35 56 90 257 301 412 555 777

    # from Jupyter
    import run_experiment
    out = run_experiment.main(["--dataset", "ieee38", "--quick"])

Outputs go to results/<dataset>_<kernel>/.
"""

import argparse
import datetime
import importlib.metadata
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd

import config


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Quantum-classical dual-kernel SVC fault classification experiment")
    p.add_argument("--dataset", default=config.SELECTED_DATASET, choices=sorted(config.VALID_DATASET_CHOICES))
    p.add_argument("--kernel", default=config.SELECTED_QUANTUM_KERNEL,
                   choices=sorted(config.VALID_QUANTUM_KERNEL_CHOICES))
    p.add_argument("--seeds", type=int, nargs="+", default=config.SEED_CANDIDATES,
                   help="Predefined seeds. Each seed is one independent data draw.")
    p.add_argument("--report-seed", default="first",
                   help="Seed for the detailed single-split report: 'first' (default, pre-declared), "
                        "an integer seed, or 'best' (largest dual advantage; biased, for reproducing "
                        "the original script only).")
    p.add_argument("--quick", action="store_true",
                   help="Tiny datasets and reduced grids for a fast end-to-end check. Not for results.")
    p.add_argument("--show", action="store_true", help="Show figures interactively (default: save only).")
    return p.parse_args(argv)


def apply_quick_mode():
    config.N_SAMPLES_VRE0_TRAIN = 60
    config.N_SAMPLES_VRE0_TEST = 40
    config.N_SAMPLES_VRE30 = 40
    config.N_SAMPLES_VRE80 = 40
    config.CLASSICAL_PARAM_GRIDS = [
        {"kernel": ["rbf"], "C": [1, 10], "gamma": [0.1, 1]},
        {"kernel": ["linear"], "C": [1, 10]},
    ]
    config.QUANTUM_ENTANGLEMENTS = ["linear"]
    config.QUANTUM_C_GRID = [1, 10]
    config.DUAL_C_GRID = [1, 10]
    config.DUAL_ALPHA_GRID = np.linspace(0, 1, 6)
    config.BOOTSTRAP_N = 1000


def environment_info():
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pandapower", "qiskit",
                "qiskit-aer", "qiskit-ibm-runtime", "qiskit-machine-learning"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions}


def run_one_seed(seed, dataset, mode, preprocessing, models, stats, reporting):
    t0 = time.time()
    split = preprocessing.prepare_vre0_split_and_vre_tests(
        seed,
        n_vre0_train=config.N_SAMPLES_VRE0_TRAIN,
        n_vre0_test=config.N_SAMPLES_VRE0_TEST,
        n_vre30=config.N_SAMPLES_VRE30,
        n_vre80=config.N_SAMPLES_VRE80,
        dataset=dataset,
    )
    classical, quantum, dual = models.tune_all(split, mode=mode, verbose=False)
    results = {"classical": classical, "quantum": quantum, "dual": dual}
    quantum.pop("cache", None)  # kernel matrices are on disk; free memory

    ci_df = stats.model_ci_table(split, results)
    delta_df = stats.dual_vs_classical_delta_table(split, results)
    flags = stats.strict_win_flags(results, delta_df)
    row = reporting.seed_summary_row(split, results, flags)
    row["elapsed_sec"] = time.time() - t0
    return {"split": split, "results": results, "ci_df": ci_df, "delta_df": delta_df, "summary": row}


def choose_reporting_seed(experiments, how):
    if how == "first":
        return experiments[0]
    if how == "best":
        def key(e):
            r = e["summary"]
            return (r["strict_all_win"], r["strict_test80_95ci_win"],
                    r["dual_acc80"] - r["classical_acc80"], r["dual_auc80"] - r["classical_auc80"],
                    r["dual_cv"] - r["classical_cv"], r["dual_acc30"] - r["classical_acc30"])
        return max(experiments, key=key)
    seed = int(how)
    for e in experiments:
        if e["summary"]["seed"] == seed:
            return e
    raise ValueError("--report-seed {} is not in --seeds".format(seed))


def main(argv=None):
    args = parse_args(argv)
    warnings.filterwarnings("ignore")

    # Configure BEFORE importing pipeline modules (some defaults are read at import).
    config.set_run_options(dataset=args.dataset, quantum_kernel=args.kernel)
    if args.quick:
        apply_quick_mode()
    config.ensure_dirs()

    import preprocessing
    import models
    import stats
    import reporting

    out_dir = config.run_results_dir()
    mode = config.QUANTUM_KERNEL_MODE
    started = datetime.datetime.now().isoformat(timespec="seconds")

    print("Dataset: {} | quantum kernel: {} | seeds: {} | quick: {}".format(
        args.dataset, mode, args.seeds, args.quick))
    print("Results folder:", out_dir)

    # ---------- repeated-seed loop ----------
    experiments, prediction_tables = [], []
    for seed in args.seeds:
        print("\nRunning seed {} ...".format(seed))
        exp = run_one_seed(seed, args.dataset, mode, preprocessing, models, stats, reporting)
        experiments.append(exp)
        prediction_tables.append(reporting.prediction_table_for_seed(exp["split"], exp["results"]))
        r = exp["summary"]
        print("  seed {seed}: CV d/c={dual_cv:.3f}/{classical_cv:.3f}  "
              "VRE0 d/c={dual_acc0:.3f}/{classical_acc0:.3f}  "
              "VRE30 d/c={dual_acc30:.3f}/{classical_acc30:.3f}  "
              "VRE80 d/c={dual_acc80:.3f}/{classical_acc80:.3f}  ({elapsed_sec:.0f}s)".format(**r))

    seed_df = pd.DataFrame([e["summary"] for e in experiments])
    summary_df = stats.repeated_seed_model_summary(seed_df)
    pairwise_df = stats.repeated_seed_pairwise_metric_tests(seed_df)
    dual_delta_df = stats.repeated_seed_dual_delta_tests(seed_df)
    cost_df = stats.model_cost_summary(seed_df)

    reporting.save_table(seed_df, "seed_summary", out_dir)
    reporting.save_table(pd.concat(prediction_tables, ignore_index=True), "seed_level_predictions", out_dir)
    reporting.save_table(summary_df, "repeated_seed_model_summary", out_dir)
    reporting.save_table(pairwise_df, "repeated_seed_all_pairwise_metric_tests", out_dir)
    reporting.save_table(dual_delta_df, "repeated_seed_dual_minus_classical_tests", out_dir)
    reporting.save_table(cost_df, "model_cost_summary", out_dir)

    reporting.show_table(summary_df, "Repeated-seed model summary (mean over seeds)")
    reporting.show_table(pairwise_df, "Repeated-seed paired tests (primary inference)")
    reporting.show_table(cost_df, "Computational cost per seed (seconds; only meaningful on a cold cache)", "{:.2f}")

    for metric in ["acc"]:
        reporting.plot_repeated_seed_summary(
            seed_df, metric=metric, show=args.show,
            save_path="{}/repeated_seed_{}.png".format(out_dir, metric))

    # ---------- detailed report for one seed ----------
    chosen = choose_reporting_seed(experiments, args.report_seed)
    split, results = chosen["split"], chosen["results"]
    tag = "reporting_seed"
    print("\nReporting seed: {} (selection: {})".format(split["seed"], args.report_seed))
    if args.report_seed == "best":
        print("NOTE: 'best' picks the seed most favourable to the dual kernel. Present the "
              "repeated-seed tables above as the main result.")

    preprocessing.describe_split(split)
    comparison_df = reporting.model_comparison_table(results)
    final_pairwise_df = stats.final_fixed_test_pairwise_table(split, results, random_state=split["seed"])

    reporting.show_table(comparison_df, "Model comparison, seed {}".format(split["seed"]))
    reporting.show_table(chosen["ci_df"], "Bootstrap 95% CIs, seed {}".format(split["seed"]))
    reporting.show_table(final_pairwise_df, "Fixed-test pairwise tests, seed {}".format(split["seed"]))

    reporting.save_table(comparison_df, tag + "_model_comparison", out_dir)
    reporting.save_table(chosen["ci_df"], tag + "_bootstrap_ci", out_dir)
    reporting.save_table(chosen["delta_df"], tag + "_dual_minus_classical", out_dir)
    reporting.save_table(final_pairwise_df, tag + "_pairwise_tests", out_dir)
    reporting.save_table(results["dual"]["results_df"].drop(columns=["fold_scores"]), tag + "_dual_grid", out_dir)
    reporting.classification_reports(split, results, out_dir=out_dir, verbose=False)
    reporting.plot_alpha_curve(split, results["dual"], show=args.show,
                               save_path="{}/{}_alpha_curve.png".format(out_dir, tag))

    # ---------- run metadata ----------
    reporting.save_json({
        "started": started,
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv if argv is None else argv,
        "args": vars(args),
        "reporting_seed": int(split["seed"]),
        "config": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in vars(config).items()
                   if k.isupper() and isinstance(v, (int, float, str, list, dict, tuple, type(None), np.ndarray))},
        "environment": environment_info(),
    }, "run_config.json", out_dir)

    print("\nAll outputs written to", out_dir)
    return {"seed_df": seed_df, "summary_df": summary_df, "pairwise_df": pairwise_df,
            "cost_df": cost_df, "reporting_experiment": chosen, "out_dir": out_dir}


if __name__ == "__main__":
    main()
