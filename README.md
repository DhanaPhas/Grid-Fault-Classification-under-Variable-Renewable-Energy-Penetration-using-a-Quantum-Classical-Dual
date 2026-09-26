# Grid Fault Classification under Variable Renewable Energy Penetration using a Quantum-Classical Dual Kernel

This repository contains the code and data-generation pipeline for the paper "Grid Fault Classification under Variable Renewable Energy Penetration using a Quantum-Classical Dual Kernel".

## Overview

This work investigates quantum-classical dual-kernel learning for classifying power-system faults under different levels of variable renewable energy (VRE) penetration. The objective is to evaluate a quantum-classical dual-kernel SVM across modified IEEE 38-, 68- and 123-bus networks, operating conditions and fault-feature representations, rather than to establish a quantum advantage over classical machine learning.

Fault currents are computed with pandapower's IEC 60909 short-circuit calculation. Quantum kernels are built with Qiskit (ideal statevector and noisy fake-backend variants), and preprocessing, training and evaluation use scikit-learn. Performance is assessed with cross-validation, held-out test accuracy, ROC AUC and repeated-seed paired statistical tests.

## Method summary

1. **Data generation** (`data/Mod_IEEE.py`). For each sample, loads are randomly scaled (0.5–1.5×), an hour of the day selects the PV/wind output, and one of four classes is simulated: three-phase (`3ph`), phase-to-phase (`2ph`) or single-phase-to-ground (`1ph`) faults at a random bus with random fault resistance, or no fault (`NF`, load flow). Features are the five largest line currents adjacent to the monitored buses plus their mean.
2. **Train/test design.** Models are trained on VRE0 (no renewables) only. They are tested on an independent VRE0 set and on held-out VRE30 and VRE80 sets, which measures robustness to renewable penetration the model never saw.
3. **Preprocessing** (`preprocessing.py`). StandardScaler → PCA (95% variance) → MinMax scaling to [0, π], fit on VRE0 training data only.
4. **Models.** Classical SVC (RBF / polynomial / linear), quantum SVC with a ZZFeatureMap fidelity kernel, and a dual kernel `K = α·K_quantum + (1 − α)·K_classical` with α selected by cross-validation.
5. **Evaluation.** Bootstrap confidence intervals, McNemar tests on a fixed test set, and paired permutation tests across predefined seeds. ROC AUC is also reported from SVC decision scores.

## Networks

All networks are modified, balanced 20 kV models defined directly in `data/Mod_IEEE.py`. They are not the standard IEEE test-feeder data files; line and load parameters, VRE placement and monitored buses are specified in that module.

| Network | Buses | Lines | Loads | Monitored buses | VRE30 units (PV + wind) | VRE80 units (PV + wind) |
|---|---|---|---|---|---|---|
| ieee38 | 38 | 37 | 17 | 7 | 7 + 5 | 15 + 16 |
| ieee68 | 68 | 80 | 35 | 15 | 13 + 12 | 29 + 28 |
| ieee123 | 123 | 122 | 60 | 35 | 20 + 18 | 44 + 43 |

VRE30 and VRE80 mean installed renewable capacity equal to 30% and 80% of the total base load.

## Repository structure

```
.
├── config.py            # all experiment settings (dataset, kernel, grids, seeds, paths)
├── data/
│   ├── __init__.py
│   └── Mod_IEEE.py      # IEEE network builders and dataset generation
├── preprocessing.py     # scaling + PCA split builder
├── kernels.py           # classical and quantum kernels, kernel caching
├── models.py            # CV tuning for classical, quantum and dual-kernel SVC
├── stats.py             # bootstrap CIs, permutation / McNemar tests
├── reporting.py         # tables, classification reports, figures
├── run_experiment.py    # command-line entry point for the full pipeline
├── requirements.txt
└── README.md
```

Each run writes to `results/<dataset>_<quantum kernel>/`.

## Installation

Tested with Python 3.10.

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

Main package versions used for the paper:

| Package | Version |
|---|---|
| pandapower | 3.3.2 |
| qiskit | 2.3.0 |
| qiskit-aer | 0.17.2 |
| qiskit-ibm-runtime | 0.45.0 |
| qiskit-machine-learning | 0.9.0 |
| scikit-learn | 1.7.2 |

Installing `numba` is optional but speeds up pandapower considerably.

## Usage

Run all commands from the repository root.

### Full experiment

```bash
# fast end-to-end check (tiny data, reduced grids; a few minutes)
python run_experiment.py --dataset ieee38 --quick

# full run
python run_experiment.py --dataset ieee38 --kernel FSK
python run_experiment.py --dataset ieee68 --kernel FQK \
    --seeds 14 22 35 56 90 257 301 412 555 777
```

Options: `--dataset {ieee38,ieee68,ieee123}`, `--kernel {FSK,FQK}`, `--seeds ...`, `--report-seed {first,best,<seed>}`, `--quick`, `--show`.

From Jupyter:

```python
import run_experiment
out = run_experiment.main(["--dataset", "ieee38", "--quick"])
```

### Outputs

| File | Content |
|---|---|
| `repeated_seed_all_pairwise_metric_tests.csv` | **Primary result.** Mean paired difference across seeds for each model pair and accuracy metric, sign-flip permutation p-value (exact for up to 20 seeds, Monte Carlo beyond that; unadjusted), bootstrap CI, t-test and Wilcoxon for reference |
| `repeated_seed_dual_minus_classical_tests.csv` | Paired accuracy differences between dual and classical models across seeds |
| `repeated_seed_model_summary.csv` | Mean, std, min and max of cross-validation and test accuracy metrics across seeds |
| `seed_summary.csv`, `seed_level_predictions.csv` | Per-seed metrics (including ROC AUC) and every individual test prediction |
| `model_cost_summary.csv` | Tuning time per seed; the "Dual (end-to-end)" row includes the classical and quantum stages the dual kernel depends on |
| `reporting_seed_*` | Detailed single-split tables: model comparison, bootstrap CIs, fixed-test pairwise tests, dual-kernel grid, and classification reports; an alpha-curve figure is also saved |
| `repeated_seed_acc.png` | Mean ± std accuracy across seeds |
| `run_config.json` | Arguments, all config values and package versions for the run |

### Statistical notes

- Claims should rest on the repeated-seed table. The single-seed tests describe one split only.
- The pairwise permutation p-values are not adjusted for multiple comparisons. Interpret them accordingly and prioritize effect sizes and confidence intervals. With *n* seeds, the smallest possible two-sided exact sign-flip p-value is 2/2^n; 10 or more seeds is advisable for more stable estimates.
- For a more statistically stable estimate, run the experiment with 30 distinct seeds. For example, on macOS/Linux:

    ```bash
    python run_experiment.py --dataset ieee68 --kernel FQK --seeds $(seq 1 30)
    ```

    Using a fixed, documented seed list makes the results reproducible; increasing the number of seeds improves the precision of the across-seed summary but does not replace independent test-set evaluation.
- The reporting seed defaults to the first seed (pre-declared). `--report-seed best` reproduces the original selection of the seed most favourable to the dual kernel and should not be used for headline results.
- Timing is only meaningful on a cold cache, because cached quantum kernels load in seconds.

### Individual modules

```python
from data.Mod_IEEE import generate_dataset
import preprocessing, models

df = generate_dataset(N=650, vre_percent=0, random_seed=14, dataset="ieee38")

split = preprocessing.prepare_vre0_split_and_vre_tests(seed=14)
preprocessing.describe_split(split)

classical, quantum, dual = models.tune_all(split)
print(models.summarize_results(classical, quantum, dual))
```

Generated datasets and quantum kernel matrices are cached in `config.CACHE_DIR` and reused on later runs.

## Citation

If you use this repository in your research, please cite:

> [Paper citation to be added]

## License

This project is released under the [MIT License](LICENSE), a permissive open-source license. You are free to use, copy, modify, merge, publish and distribute the code, including for reproducing, verifying or extending the results reported in the paper, provided the original copyright notice and license text are included in any copies or substantial portions of the software.

The code and data-generation pipeline are made openly available to support the reproducibility of the published results, in line with the open-science and reproducibility expectations of IEEE Access. If you use this code in academic work, please cite the paper (see [Citation](#citation)).
