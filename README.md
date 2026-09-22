# Grid Fault Classification under Variable Renewable Energy Penetration using a Quantum-Classical Dual Kernel

This repository contains the code and data generation pipeline in the paper "Grid Fault Classification under Variable Renewable Energy Penetration using a Quantum-Classical Dual Kernel".

## Overview

This work investigates the use of quantum-classical dual-kernel learning for classifying power-system faults under varying renewable energy penetration levels. The objective is evaluate the quantum-classical dual kernel SVM across different IEEE multi-bus network scales (modified IEEE 38-, 68, and 123-bus), operating conditions, and fault-feature representations rather than to establish a quantum advantage over classical machine learning. The algorithmic pipeline is implemented using IBM \texttt{qiskit} packages and \texttt{scikit-learn} for quantum kernel construction, data preprocessing, model training, and evaluation. Model performance is assessed using cross-validation and held-out test accuracy, with the noisy backend evaluation applied to the IEC 60909 pandapower experiment.

## Some Required/Used Packages

- python                       3.10.19
- pandapower                   3.3.2
- qiskit                       2.3.0
- qiskit-aer                   0.17.2
- qiskit-ibm-runtime           0.45.0
- qiskit-machine-learning      0.9.0
- scikit-learn                 1.7.2

## Repository Structure

- `data/` — Generated datasets
- `src/` — Source code
- `notebooks/` — Jupyter notebooks
- `results/` — Experimental results
- `README.md` — Project documentation




## Usage

Instructions for reproducing the experiments will be provided here.

## Citation

If you use this repository in your research, please cite:

> [Your paper citation here]
