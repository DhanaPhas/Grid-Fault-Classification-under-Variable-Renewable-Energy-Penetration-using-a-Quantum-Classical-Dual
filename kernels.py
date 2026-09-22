"""
kernels.py

Classical kernel matrices, the quantum kernel factory, and on-disk caching of
precomputed quantum kernel matrices.

Qiskit is imported lazily inside the quantum functions, so classical-only
code (and the dual kernel's classical half) can import this module without
Qiskit installed.
"""

import hashlib
import json
import os

import numpy as np
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel

import config


# =========================
# Classical kernels
# =========================

def classical_kernel_matrix(X, Y, kernel_type, gamma=0.1, degree=3, coef0=1):
    """Gram matrix K[i, j] = k(X[i], Y[j]) for the SVC kernel types used in tuning."""
    if kernel_type == "rbf":
        return rbf_kernel(X, Y, gamma=gamma)
    if kernel_type == "poly":
        return polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)
    if kernel_type == "linear":
        return linear_kernel(X, Y)
    raise ValueError("Unknown classical kernel type: {}".format(kernel_type))


def classical_kernel_from_params(X, Y, params):
    """Build a classical Gram matrix from a tuned SVC params dict."""
    return classical_kernel_matrix(
        X,
        Y,
        params.get("kernel", "rbf"),
        gamma=params.get("gamma", 0.1),
        degree=params.get("degree", 3),
        coef0=params.get("coef0", 1),
    )


# =========================
# Quantum kernels
# =========================

def _load_fake_backend(name=config.NOISY_FAKE_BACKEND):
    try:
        from qiskit_ibm_runtime import fake_provider
    except Exception as exc:
        raise ImportError(
            "The noisy fidelityquantumkernel mode needs qiskit-ibm-runtime. "
            "Import error: {}".format(exc)
        )
    if not hasattr(fake_provider, name):
        raise ValueError("Fake backend {!r} not found in qiskit_ibm_runtime.fake_provider".format(name))
    return getattr(fake_provider, name)()


def make_quantum_kernel(
    feature_dim,
    reps=config.QUANTUM_FEATUREMAP_REPS,
    entanglement="linear",
    mode=config.QUANTUM_KERNEL_MODE,
):
    """
    ZZFeatureMap fidelity kernel.

    mode="fidelitystatevectorkernel": exact, noiseless statevector simulation.
    mode="fidelityquantumkernel":     ComputeUncompute sampling on a fake noisy backend.
    """
    from qiskit.circuit.library import ZZFeatureMap

    feature_map = ZZFeatureMap(
        feature_dimension=feature_dim,
        reps=reps,
        entanglement=entanglement,
        parameter_prefix=config.QUANTUM_FEATUREMAP_PARAMETER_PREFIX,
    )

    mode = mode.lower()
    if mode == "fidelitystatevectorkernel":
        from qiskit_machine_learning.kernels import FidelityStatevectorKernel

        return FidelityStatevectorKernel(feature_map=feature_map)

    if mode == "fidelityquantumkernel":
        try:
            from qiskit_ibm_runtime import SamplerV2
            from qiskit_machine_learning.kernels import FidelityQuantumKernel
            from qiskit_machine_learning.state_fidelities import ComputeUncompute
        except Exception as exc:
            raise ImportError(
                "fidelityquantumkernel mode needs qiskit-machine-learning state_fidelities "
                "and qiskit-ibm-runtime. Import error: {}".format(exc)
            )
        sampler = SamplerV2(_load_fake_backend())
        fidelity = ComputeUncompute(sampler=sampler)
        return FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

    raise ValueError("Unknown quantum kernel mode: {}".format(mode))


# =========================
# Kernel cache
# =========================

def array_digest(*arrays):
    """SHA-256 over shapes, dtypes and bytes, so the cache key changes if the data changes."""
    digest = hashlib.sha256()
    for arr in arrays:
        arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def quantum_kernel_cache_path(X_train, X_test0, X_test30, X_test80, entanglement, reps, mode):
    """
    Cache file path and key. The key format matches the original pipeline,
    so previously saved .npz files are reused.
    """
    key_payload = {
        "pipeline_tag": config.CACHE_PIPELINE_TAG,
        "dataset": config.SELECTED_DATASET,
        "quantum_kernel_mode": mode.lower(),
        "featuremap": "ZZFeatureMap",
        "feature_dimension": int(X_train.shape[1]),
        "reps": int(reps),
        "entanglement": entanglement,
        "parameter_prefix": config.QUANTUM_FEATUREMAP_PARAMETER_PREFIX,
        "array_digest": array_digest(X_train, X_test0, X_test30, X_test80),
    }
    key_text = json.dumps(key_payload, sort_keys=True)
    key_hash = hashlib.sha256(key_text.encode("utf-8")).hexdigest()[:24]
    filename = "qkernel_{}_{}_ent{}_reps{}_{}.npz".format(
        config.SELECTED_DATASET,
        mode.lower(),
        entanglement,
        reps,
        key_hash,
    )
    return os.path.join(config.CACHE_DIR, filename), key_payload


def load_cached_quantum_kernels(cache_path):
    with np.load(cache_path, allow_pickle=False) as data:
        return data["K_train"], data["K_test0"], data["K_test30"], data["K_test80"]


def save_cached_quantum_kernels(cache_path, key_payload, K_train, K_test0, K_test30, K_test80):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp_cache_path = "{}.tmp.{}.npz".format(cache_path, os.getpid())
    np.savez_compressed(
        tmp_cache_path,
        K_train=K_train,
        K_test0=K_test0,
        K_test30=K_test30,
        K_test80=K_test80,
        metadata_json=json.dumps(key_payload, sort_keys=True),
    )
    os.replace(tmp_cache_path, cache_path)


def precompute_quantum_kernels(
    X_train,
    X_test0,
    X_test30,
    X_test80,
    entanglement,
    reps=config.QUANTUM_FEATUREMAP_REPS,
    mode=config.QUANTUM_KERNEL_MODE,
    use_cache=config.USE_PRECOMPUTED_KERNEL_CACHE,
):
    """
    Returns (K_train, K_test0, K_test30, K_test80, qkernel).
    Test matrices have shape (n_test, n_train), as SVC(kernel="precomputed") expects.
    qkernel is None when the matrices came from the cache.
    """
    mode = mode.lower()
    cache_path, key_payload = quantum_kernel_cache_path(
        X_train, X_test0, X_test30, X_test80, entanglement, reps, mode
    )

    if use_cache and os.path.exists(cache_path):
        print("  loading cached quantum kernels:", cache_path)
        K_train, K_test0, K_test30, K_test80 = load_cached_quantum_kernels(cache_path)
        return K_train, K_test0, K_test30, K_test80, None

    qkernel = make_quantum_kernel(X_train.shape[1], reps=reps, entanglement=entanglement, mode=mode)
    K_train = qkernel.evaluate(X_train, X_train)
    K_test0 = qkernel.evaluate(X_test0, X_train)
    K_test30 = qkernel.evaluate(X_test30, X_train)
    K_test80 = qkernel.evaluate(X_test80, X_train)

    if use_cache:
        save_cached_quantum_kernels(cache_path, key_payload, K_train, K_test0, K_test30, K_test80)
        print("  saved cached quantum kernels:", cache_path)

    return K_train, K_test0, K_test30, K_test80, qkernel


# =========================
# Dual kernel
# =========================

def combine_kernels(K_quantum, K_classical, alpha):
    """Convex combination alpha * K_q + (1 - alpha) * K_c. alpha is the quantum weight."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1], got {}".format(alpha))
    return alpha * K_quantum + (1.0 - alpha) * K_classical
