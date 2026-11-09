from itertools import product
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from joblib import Parallel, delayed

from ..data.load_data import load_data
from ..data.prepare_data import (
    compute_train_stats,
    build_exog_features,
    build_train_base_direct,
    build_train_base_recursive,
    build_validation_base_direct,
    build_validation_base_recursive,
    expand_train_horizons,
    add_horizon_to_validation,
)
from ..config import SERIE, OUTPUT_DIR_TREE
from .models_utils import (
    tree_and_boosting_pipeline_direct_2,
    tree_and_boosting_recursive_pipeline_2,
    tree_and_boosting_predict_and_report,
    tree_and_boosting_recursive_pipeline,
)

# --- 1. Shared search spaces ------------------------------------------------
# Both strategies sit in the SAME grid; "estrategia" is just one axis.
BOOSTING_GRID = {
    "estrategia": ["direto", "recursivo"],
    "max_iter": [200, 400],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 4],
    "min_samples_leaf": [10, 20],
    "l2_regularization": [0.0, 0.5, 1.5],
}

ARVORE_GRID = {
    "estrategia": ["direto", "recursivo"],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [10, 20, 40],
    "ccp_alpha": [0.0, 1e-4]
}

# --- 2. Helpers -------------------------------------------------------------
def expand_grid(grid):
    keys = list(grid)
    for values in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))

def build_estimator(class_name, config):
    params = {k: v for k, v in config.items() if k != "estrategia"}
    if class_name == "boosting":
        # early_stopping=False avoids sklearn's *random* internal split,
        # which would violate the temporal spirit even inside the training set.
        return HistGradientBoostingRegressor(
            random_state=0, early_stopping=False, **params
        )
    if class_name == "arvore":
        return DecisionTreeRegressor(random_state=0, **params)
    raise ValueError(class_name)

# --- 3. Walk-forward folds inside the training window -----------------------
def make_folds(train_end, n_folds=3, horizon=28):
    """
    Non-overlapping validation blocks; the most recent one ends at `train_end`.
    Example with train_end=2016-03-27, horizon=28, n_folds=3:
      fold 0: valid 2016-02-28 .. 2016-03-27,  train ≤ 2016-02-27
      fold 1: valid 2016-01-31 .. 2016-02-27,  train ≤ 2016-01-30
      fold 2: valid 2016-01-03 .. 2016-01-30,  train ≤ 2016-01-02
    """
    folds = []
    for k in range(n_folds):
        valid_end      = train_end - pd.Timedelta(days=k * horizon)
        valid_start    = valid_end - pd.Timedelta(days=horizon - 1)
        train_fold_end = valid_start - pd.Timedelta(days=1)
        folds.append((train_fold_end, valid_start, valid_end))
    return folds


def seasonal_naive_scale(train_df, series, period=7):
    """In-sample MAE of seasonal naive on this fold's train set."""
    y = train_df[series].to_numpy()
    return float(np.mean(np.abs(y[period:] - y[:-period])))

def prepare_fold_direct(
    train_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
):
    """
    Mirror of get_all_data's internal steps, restricted to one fold.

    - train_fold: rows of train_df with date ≤ train_end
    - val_fold: rows of train_df with valid_start ≤ date ≤ valid_end (the fold's validation lives *inside* the training window; the official val_df is never used for CV)
    - stats: computed from train_fold and cov restricted to ≤ train_end (strictly train-only, per the enunciado)
    - exog: covariates up to valid_end — prices/events/SNAP are known at the fold's origin by construction
    """
    train_fold = train_df[train_df["date"] <= train_end].copy()
    val_fold = train_df[
        (train_df["date"] >= valid_start) & (train_df["date"] <= valid_end)
    ].copy()

    cov_for_stats = cov_df[cov_df["date"] <= train_end]
    cov_for_exog = cov_df[cov_df["date"] <= valid_end]

    # 1) Train-only statistics
    stats_fold = compute_train_stats(train_fold, cov_for_stats)

    # 2) Exogenous features (standardized with fold-train stats)
    exog_feats_fold = build_exog_features(cov_for_exog, stats_fold)

    # 3) Train / validation bases (validation uses train lags only)
    X_train_df, y_train_std = build_train_base_direct(
        train_fold, exog_feats_fold, stats_fold
    )
    X_validation = build_validation_base_direct(
        exog_feats_fold, y_train_std, stats_fold, val_fold.index # type: ignore
    )

    # 4) Direct horizon expansion
    X_train_h, y_train_h = expand_train_horizons(X_train_df, y_train_std)
    X_val_h = add_horizon_to_validation(X_validation)

    val_fold = val_fold.drop(columns=['date'])

    return X_train_h, y_train_h, X_val_h, stats_fold, val_fold

def prepare_fold_recursive(
    train_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
):
    """
    Mirror of get_all_data's internal steps, restricted to one fold.

    - train_fold: rows of train_df with date ≤ train_end
    - val_fold: rows of train_df with valid_start ≤ date ≤ valid_end (the fold's validation lives *inside* the training window; the official val_df is never used for CV)
    - stats: computed from train_fold and cov restricted to ≤ train_end (strictly train-only, per the enunciado)
    - exog: covariates up to valid_end — prices/events/SNAP are known at the fold's origin by construction
    """
    train_fold = train_df[train_df["date"] <= train_end].copy()
    val_fold = train_df[
        (train_df["date"] >= valid_start) & (train_df["date"] <= valid_end)
    ].copy()

    cov_for_stats = cov_df[cov_df["date"] <= train_end]
    cov_for_exog = cov_df[cov_df["date"] <= valid_end]

    # 1) Train-only statistics
    stats_fold = compute_train_stats(train_fold, cov_for_stats)

    # 2) Exogenous features (standardized with fold-train stats)
    exog_feats_fold = build_exog_features(cov_for_exog, stats_fold)

    # 3) Train / validation bases (validation uses train lags only)
    X_train_df, y_train_std = build_train_base_recursive(
        train_fold, exog_feats_fold, stats_fold
    )
    X_validation = build_validation_base_recursive(
        exog_feats_fold, stats_fold, val_fold.index # type: ignore
    )

    # 4) Direct horizon expansion
    X_train_h, y_train_h = expand_train_horizons(X_train_df, y_train_std)
    X_val_h = add_horizon_to_validation(X_validation)

    val_fold = val_fold.drop(columns=['date'])

    return X_train_h, y_train_h, X_val_h, stats_fold, val_fold

def prepare_folds_direct(
    train_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    serie: str,
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
) -> list[dict]:
    """
    Build X_train_h, y_train_h, X_val_h, stats, y_true, scale once per fold.
    All inputs derive from train_df / cov_df restricted to ≤ each fold's
    train_end, so no fold touches the official validation window.
    """
    prepared = []
    for k, (train_end, valid_start, valid_end) in enumerate(folds):
        X_train_h, y_train_h, X_val_h, stats_fold, val_fold = prepare_fold_direct(
            train_df, cov_df, train_end, valid_start, valid_end
        )

        val_fold = val_fold.sort_values("date")
        y_true = val_fold[serie].to_numpy()

        # Seasonal-naive scale from fold-train only (enunciado's MASE definition)
        train_fold = train_df[train_df["date"] <= train_end]
        y_tr = train_fold[serie].to_numpy()
        scale = float(np.mean(np.abs(y_tr[7:] - y_tr[:-7])))

        prepared.append({
            "fold_idx":   k,
            "train_end":  train_end,
            "valid_start": valid_start,
            "valid_end":  valid_end,
            "X_train_h":  X_train_h,
            "y_train_h":  y_train_h,
            "X_val_h":    X_val_h,
            "stats":      stats_fold,
            "y_true":     y_true,
            "scale":      scale,
        })
    return prepared

def prepare_folds_recursive(
    train_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    serie: str,
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
) -> list[dict]:
    """
    Build X_train_h, y_train_h, X_val_h, stats, y_true, scale once per fold.
    All inputs derive from train_df / cov_df restricted to ≤ each fold's
    train_end, so no fold touches the official validation window.
    """
    prepared = []
    for k, (train_end, valid_start, valid_end) in enumerate(folds):
        X_train, y_train, X_val, stats_fold, val_fold = prepare_fold_recursive(
            train_df, cov_df, train_end, valid_start, valid_end
        )

        val_fold = val_fold.sort_values("date")
        y_true = val_fold[serie].to_numpy()

        # Seasonal-naive scale from fold-train only (enunciado's MASE definition)
        train_fold = train_df[train_df["date"] <= train_end]
        y_tr = train_fold[serie].to_numpy()
        scale = float(np.mean(np.abs(y_tr[7:] - y_tr[:-7])))

        prepared.append({
            "fold_idx": k,
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": valid_end,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "stats": stats_fold,
            "y_true": y_true,
            "scale": scale,
        })
    return prepared

def evaluate_config_recursive(
    class_name: str,
    config: dict,
    prepared_folds: list[dict],
) -> tuple[float, dict]:
    """Return (mean MASE across folds, config). One fresh model per fold."""
    mases = []
    for pf in prepared_folds:
        model = build_estimator(class_name, config)
        y_hat = tree_and_boosting_recursive_pipeline_2(
            model,
            model_name=f"{class_name}[h{config.get('max_depth','?')}]",
            X_train=pf["X_train"],
            y_train=pf["y_train"],
            X_val=pf["X_val"],
            stats=pf["stats"],
        )

        y_true = pf["y_true"]
        y_hat = np.asarray(y_hat).ravel()
        if y_true.shape != y_hat.shape:
            raise ValueError(
                f"shape mismatch on fold {pf['fold_idx']}: "
                f"y_true {y_true.shape} vs y_hat {y_hat.shape}"
            )
        mae = float(np.mean(np.abs(y_true - y_hat)))
        mases.append(mae / pf["scale"])

    return float(np.mean(mases)), config

def evaluate_config_direct(
    class_name: str,
    config: dict,
    prepared_folds: list[dict],
) -> tuple[float, dict]:
    """Return (mean MASE across folds, config). One fresh model per fold."""
    mases = []
    for pf in prepared_folds:
        model = build_estimator(class_name, config)
        y_hat = tree_and_boosting_pipeline_direct_2(
            model,
            model_name=f"{class_name}[h{config.get('max_depth','?')}]",
            X_train_h=pf["X_train_h"],
            y_train_h=pf["y_train_h"],
            X_val_h=pf["X_val_h"],
            stats=pf["stats"],
        )

        y_true = pf["y_true"]
        y_hat = np.asarray(y_hat).ravel()
        if y_true.shape != y_hat.shape:
            raise ValueError(
                f"shape mismatch on fold {pf['fold_idx']}: "
                f"y_true {y_true.shape} vs y_hat {y_hat.shape}"
            )
        mae = float(np.mean(np.abs(y_true - y_hat)))
        mases.append(mae / pf["scale"])

    return float(np.mean(mases)), config

def grid_search_recursive(
    class_name: str,
    grid: dict,
    prepared_folds: list[dict],
    n_jobs: int = -1,
) -> list[tuple[float, dict]]:
    """
    Run the whole grid for the direct strategy on precomputed folds.
    Returns configs sorted by ascending mean MASE (best first).
    """
    grid = {**grid, 'estrategia': ['recursivo']}
    
    configs = list(expand_grid(grid))
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(evaluate_config_recursive)(class_name, cfg, prepared_folds)
        for cfg in configs
    )
    results.sort(key=lambda t: t[0]) # type: ignore
    
    return results # type: ignore

def grid_search_direct(
    class_name: str,
    grid: dict,
    prepared_folds: list[dict],
    n_jobs: int = -1,
) -> list[tuple[float, dict]]:
    """
    Run the whole grid for the direct strategy on precomputed folds.
    Returns configs sorted by ascending mean MASE (best first).
    """
    grid = {**grid, 'estrategia': ['direto']}
    
    configs = list(expand_grid(grid))
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(evaluate_config_direct)(class_name, cfg, prepared_folds)
        for cfg in configs
    )
    results.sort(key=lambda t: t[0]) # type: ignore
    
    return results # type: ignore

# --- 7. Main ---------------------------------------------------------------
if __name__ == "__main__":
    train_df, val_df, cov_df = load_data()
    ORIGIN = pd.Timestamp("2016-03-27")
    FOLDS = make_folds(ORIGIN, n_folds=3, horizon=28)

    final_cv = {}
    for class_name, grid in [("arvore", ARVORE_GRID), ("boosting", BOOSTING_GRID)]:
        for estrategia, prepare, search in [
            ("direto", prepare_folds_direct, grid_search_direct),
            ("recursivo", prepare_folds_recursive, grid_search_recursive),
        ]:
            prepared = prepare(train_df, cov_df, SERIE, FOLDS)
            ranked = search(class_name, grid, prepared)
            best_mase, best_cfg = ranked[0]
            print(f"[{SERIE} | {class_name} | {estrategia}] "
                  f"CV-MASE={best_mase:.4f} | {best_cfg}")
            if class_name not in final_cv or best_mase < final_cv[class_name]["cv_mase"]:
                final_cv[class_name] = {
                    "cv_mase": best_mase,
                    "estrategia": estrategia,
                    "config": dict(best_cfg),
                }

    print("DICIONÁRIO DE PARÂMETROS FINAL")
    print(final_cv)

    estimadores = {"arvore": DecisionTreeRegressor, "boosting": HistGradientBoostingRegressor}
    linhas = []
    for class_name, res in final_cv.items():
        params = {k: v for k, v in res["config"].items() if k != "estrategia"}
        final_model = build_estimator(class_name, params)
        if res["estrategia"] == "direto":
            tree_and_boosting_predict_and_report(final_model, class_name, output_dir=OUTPUT_DIR_TREE)
        else:
            tree_and_boosting_recursive_pipeline(final_model, class_name, output_dir=OUTPUT_DIR_TREE)
        linhas.append({"series": SERIE, "modelo": class_name, "estrategia": res["estrategia"],
                       "cv_mase": res["cv_mase"], **params})
    pd.DataFrame(linhas).to_csv(OUTPUT_DIR_TREE / "cv_store_total.csv", index=False)
