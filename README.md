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
5. **Evaluation.** Bootstrap confidence intervals, McNemar tests on a fixed test set, and paired permutation tests across predefined seeds with Holm correction.

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
├── requirements.txt
└── README.md
```

Planned (in progress): `models.py` (tuning), `stats.py` (statistical tests), `reporting.py` (tables and figures), `run_experiment.py` (entry point) and a `results/` folder.

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

Generate a dataset:

```python
from data.Mod_IEEE import generate_dataset

df = generate_dataset(N=650, vre_percent=0, random_seed=14, dataset="ieee38")
```

Build the full train/test split used by the models:

```python
import preprocessing

split = preprocessing.prepare_vre0_split_and_vre_tests(seed=14)
preprocessing.describe_split(split)
```

The network and quantum-kernel mode are selected in `config.py` (`SELECTED_DATASET`, `SELECTED_QUANTUM_KERNEL`). Generated datasets and quantum kernel matrices are cached in `config.CACHE_DIR` and reused on later runs.

Full experiment reproduction instructions will be added with `run_experiment.py`.

## Citation

If you use this repository in your research, please cite:

> [Paper citation to be added]

## License

[License to be added]
