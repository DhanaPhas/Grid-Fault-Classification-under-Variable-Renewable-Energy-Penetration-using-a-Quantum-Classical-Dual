"""
config.py

Central settings for the pandapower + classical / quantum / dual-kernel SVC
pipeline. Other modules should import values from here instead of defining
their own globals, e.g.

    import config
    df = generate_dataset(
        N=config.N_SAMPLES_VRE0_TRAIN,
        vre_percent=0,
        random_seed=seed,
        dataset=config.SELECTED_DATASET,
        cache_dir=config.CACHE_DIR,
        cache_tag=config.CACHE_PIPELINE_TAG,
    )
"""

import os

import numpy as np


# =========================
# Dataset and quantum kernel selection
# =========================

# Supported datasets: "ieee38", "ieee68", "ieee123".
SELECTED_DATASET = "ieee38"

# Supported quantum kernels:
#   "fidelitystatevectorkernel" - ideal (noiseless) statevector kernel
#   "fidelityquantumkernel"     - ComputeUncompute with the fake noisy backend
SELECTED_QUANTUM_KERNEL = "fidelitystatevectorkernel"

VALID_DATASET_CHOICES = {"ieee38", "ieee68", "ieee123"}
VALID_QUANTUM_KERNEL_CHOICES = {"fidelitystatevectorkernel", "fidelityquantumkernel"}

# Fake backend used when SELECTED_QUANTUM_KERNEL == "fidelityquantumkernel".
NOISY_FAKE_BACKEND = "FakeBrooklynV2"


# =========================
# Dataset sizes
# =========================

# Full-size values used for reported results.
NGEN_VRE0_TRAIN = 650
NGEN_VRE0_TEST = 400
NGEN_VRE30 = 500
NGEN_VRE80 = 600

# Values actually used by the run. For a fast end-to-end check, temporarily
# set these to something small (e.g. 160), then restore the NGEN_* values.
N_SAMPLES_VRE0_TRAIN = NGEN_VRE0_TRAIN
N_SAMPLES_VRE0_TEST = NGEN_VRE0_TEST
N_SAMPLES_VRE30 = NGEN_VRE30
N_SAMPLES_VRE80 = NGEN_VRE80

# The VRE0 test set is generated with seed + this offset so it never
# overlaps the VRE0 training draw.
VRE0_TEST_SEED_OFFSET = 100000


# =========================
# Preprocessing
# =========================

PCA_VARIANCE = 0.95                 # keep components explaining 95% of variance
ANGLE_FEATURE_RANGE = (0, np.pi)    # MinMax range before quantum encoding


# =========================
# Seeds
# =========================

# Predefined seed set. Each seed creates one train/test split reused by all models.
SEED_CANDIDATES = [14, 22, 35, 56, 90, 257]
STOP_AFTER_FIRST_STRICT_WIN = False


# =========================
# Hyperparameter grids
# =========================

CLASSICAL_PARAM_GRIDS = [
    {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1]},
    {
        "kernel": ["poly"],
        "C": [0.1, 1, 10, 100],
        "gamma": [0.001, 0.01, 0.1, 1],
        "degree": [2, 3, 4, 5],
        "coef0": [0, 1],
    },
    {"kernel": ["linear"], "C": [0.1, 1, 10, 100]},
]

QUANTUM_FEATUREMAP_REPS = 2
QUANTUM_ENTANGLEMENTS = ["linear", "circular", "full"]
QUANTUM_FEATUREMAP_PARAMETER_PREFIX = "a"
QUANTUM_C_GRID = [0.1, 1, 10, 100]

DUAL_C_GRID = [0.1, 1, 10, 100]
DUAL_ALPHA_GRID = np.linspace(0, 1, 21)  # alpha = quantum kernel weight


# =========================
# Evaluation and statistics
# =========================

CV_FOLDS = 5
BOOTSTRAP_N = 10000
BOOTSTRAP_CI = 0.95
SIGNIFICANCE_LEVEL = 0.05


# =========================
# Reporting
# =========================

SHOW_CONFUSION_MATRICES = True
CONFUSION_MATRIX_NORMALIZE = None  # None for counts, "true" for row-normalized rates.


# =========================
# Caching and output paths
# =========================

USE_DATASET_CACHE = True
USE_PRECOMPUTED_KERNEL_CACHE = True

CACHE_DIR = "pandapower_kernel_cache_conf_2_5"

# Root folder for result tables and figures. run_experiment.py writes each
# run into RESULTS_DIR/<dataset>_<quantum kernel>/.
RESULTS_DIR = "results"


def make_cache_pipeline_tag(dataset, quantum_kernel):
    """Cache-name prefix. Matches the original pipeline so old caches are reused."""
    return "{}_{}_no_split_vre0_train650_test400_v2".format(dataset.lower(), quantum_kernel.lower())


# =========================
# Validation and derived values
# =========================

SELECTED_DATASET = SELECTED_DATASET.lower()
SELECTED_QUANTUM_KERNEL = SELECTED_QUANTUM_KERNEL.lower()

if SELECTED_DATASET not in VALID_DATASET_CHOICES:
    raise ValueError("SELECTED_DATASET must be one of {}".format(sorted(VALID_DATASET_CHOICES)))
if SELECTED_QUANTUM_KERNEL not in VALID_QUANTUM_KERNEL_CHOICES:
    raise ValueError("SELECTED_QUANTUM_KERNEL must be one of {}".format(sorted(VALID_QUANTUM_KERNEL_CHOICES)))

# Backward-compatible alias used in result tables and kernel cache names.
QUANTUM_KERNEL_MODE = SELECTED_QUANTUM_KERNEL

CACHE_PIPELINE_TAG = make_cache_pipeline_tag(SELECTED_DATASET, SELECTED_QUANTUM_KERNEL)


def ensure_dirs():
    """Create cache and results folders. Call once at the start of a run."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(run_results_dir(), exist_ok=True)


def set_run_options(dataset=None, quantum_kernel=None):
    """
    Change dataset / quantum kernel at runtime and refresh derived values.

    Call this BEFORE importing preprocessing, kernels or models: some of their
    function defaults are read from config when they are imported.
    """
    global SELECTED_DATASET, SELECTED_QUANTUM_KERNEL, QUANTUM_KERNEL_MODE, CACHE_PIPELINE_TAG
    if dataset is not None:
        dataset = dataset.lower()
        if dataset not in VALID_DATASET_CHOICES:
            raise ValueError("dataset must be one of {}".format(sorted(VALID_DATASET_CHOICES)))
        SELECTED_DATASET = dataset
    if quantum_kernel is not None:
        quantum_kernel = quantum_kernel.lower()
        if quantum_kernel not in VALID_QUANTUM_KERNEL_CHOICES:
            raise ValueError("quantum_kernel must be one of {}".format(sorted(VALID_QUANTUM_KERNEL_CHOICES)))
        SELECTED_QUANTUM_KERNEL = quantum_kernel
    QUANTUM_KERNEL_MODE = SELECTED_QUANTUM_KERNEL
    CACHE_PIPELINE_TAG = make_cache_pipeline_tag(SELECTED_DATASET, SELECTED_QUANTUM_KERNEL)


def run_results_dir():
    """Results folder for the current dataset / kernel combination."""
    return os.path.join(RESULTS_DIR, "{}_{}".format(SELECTED_DATASET, SELECTED_QUANTUM_KERNEL))
