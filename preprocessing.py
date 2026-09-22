"""
preprocessing.py

Builds the train/test split used by every model:

    VRE0 train  -> fit StandardScaler -> PCA -> MinMax(0, pi)
    VRE0 test   -> transform only (internal test, same network conditions)
    VRE30 test  -> transform only (held-out, 30% renewable penetration)
    VRE80 test  -> transform only (held-out, 80% renewable penetration)

All transforms are fit on the VRE0 training set only, so no information from
any test set leaks into preprocessing.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config
from data.Mod_IEEE import generate_dataset, monitored_feature_columns


def load_datasets(
    seed,
    n_vre0_train=config.N_SAMPLES_VRE0_TRAIN,
    n_vre0_test=config.N_SAMPLES_VRE0_TEST,
    n_vre30=config.N_SAMPLES_VRE30,
    n_vre80=config.N_SAMPLES_VRE80,
    dataset=config.SELECTED_DATASET,
):
    """Generate (or load cached) raw datasets for one seed."""
    common = {
        "use_cache": config.USE_DATASET_CACHE,
        "dataset": dataset,
        "cache_dir": config.CACHE_DIR,
        "cache_tag": config.CACHE_PIPELINE_TAG,
    }
    return {
        "df0_train": generate_dataset(N=n_vre0_train, vre_percent=0, random_seed=seed, **common),
        "df0_test": generate_dataset(
            N=n_vre0_test,
            vre_percent=0,
            random_seed=seed + config.VRE0_TEST_SEED_OFFSET,
            **common,
        ),
        "df30": generate_dataset(N=n_vre30, vre_percent=30, random_seed=seed, **common),
        "df80": generate_dataset(N=n_vre80, vre_percent=80, random_seed=seed, **common),
    }


def fit_transformers(X_train_raw, seed):
    """Fit StandardScaler -> PCA -> MinMax on training data only."""
    std_scaler = StandardScaler()
    pca = PCA(n_components=config.PCA_VARIANCE, random_state=seed)
    minmax_scaler = MinMaxScaler(feature_range=config.ANGLE_FEATURE_RANGE)

    X_std = std_scaler.fit_transform(X_train_raw)
    X_pca = pca.fit_transform(X_std)
    X_train = minmax_scaler.fit_transform(X_pca)

    transformers = {"std_scaler": std_scaler, "pca": pca, "minmax_scaler": minmax_scaler}
    return X_train, transformers


def apply_transformers(X_raw, transformers):
    """Apply already-fitted transformers to new raw features."""
    X_std = transformers["std_scaler"].transform(X_raw)
    X_pca = transformers["pca"].transform(X_std)
    return transformers["minmax_scaler"].transform(X_pca)


def prepare_vre0_split_and_vre_tests(
    seed,
    n_vre0_train=config.N_SAMPLES_VRE0_TRAIN,
    n_vre0_test=config.N_SAMPLES_VRE0_TEST,
    n_vre30=config.N_SAMPLES_VRE30,
    n_vre80=config.N_SAMPLES_VRE80,
    dataset=config.SELECTED_DATASET,
):
    """
    Build the full split for one seed. Returns a dict with the same keys as
    the original pipeline, so tune_classical / tune_quantum / tune_dual work
    unchanged.
    """
    dfs = load_datasets(
        seed,
        n_vre0_train=n_vre0_train,
        n_vre0_test=n_vre0_test,
        n_vre30=n_vre30,
        n_vre80=n_vre80,
        dataset=dataset,
    )
    df0_train, df0_test, df30, df80 = dfs["df0_train"], dfs["df0_test"], dfs["df30"], dfs["df80"]

    feature_cols = monitored_feature_columns(include_hour=False)

    X_train, transformers = fit_transformers(df0_train[feature_cols].values, seed)
    X_test0 = apply_transformers(df0_test[feature_cols].values, transformers)
    X_test30 = apply_transformers(df30[feature_cols].values, transformers)
    X_test80 = apply_transformers(df80[feature_cols].values, transformers)

    return {
        "seed": seed,
        "feature_cols": feature_cols,
        "df0": df0_train,  # backward-compatible alias
        "df0_train": df0_train,
        "df0_test": df0_test,
        "df30": df30,
        "df80": df80,
        "X_train": X_train,
        "X_test0": X_test0,
        "X_test30": X_test30,
        "X_test80": X_test80,
        "y_train": df0_train["label"].values,
        "y_test0": df0_test["label"].values,
        "y_test30": df30["label"].values,
        "y_test80": df80["label"].values,
        "pca_features": X_train.shape[1],
        **transformers,
    }


def describe_split(split):
    """Print a short summary: sizes, PCA dimension, explained variance, class balance."""
    pca = split["pca"]
    print("Seed {}: PCA kept {} components ({:.1%} variance)".format(
        split["seed"], split["pca_features"], float(np.sum(pca.explained_variance_ratio_))
    ))
    for name, y_key, x_key in [
        ("VRE0 train", "y_train", "X_train"),
        ("VRE0 test", "y_test0", "X_test0"),
        ("VRE30 test", "y_test30", "X_test30"),
        ("VRE80 test", "y_test80", "X_test80"),
    ]:
        labels, counts = np.unique(split[y_key], return_counts=True)
        balance = ", ".join("{}={}".format(l, c) for l, c in zip(labels, counts))
        print("  {:<11} n={:<4} {}".format(name, split[x_key].shape[0], balance))
