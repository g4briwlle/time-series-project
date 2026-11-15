import numpy as np
import pandas as pd

import itertools
from typing import Dict
import traceback

from ...data.load_data import load_data
from ...data.prepare_data import (
    compute_train_stats,
    build_exog_features,
    build_train_base_direct,
    build_validation_base_direct,
    build_train_base_recursive,
    build_validation_base_recursive,
    expand_train_horizons,
    add_horizon_to_validation
)
from ..models_utils import predict_model, compute_metrics, _normalize_dates
from ...config import (
    SERIE, VAL_DATES
)
from .direct_mlp_class import DirectMLPForecaster
from .recursive_mlp_class import RecursiveMLPForecaster


# -------------------------------------------------------------------
# 1) Expanding-window CV folds (time-series, no leakage)
# -------------------------------------------------------------------
import numpy as np
import pandas as pd
import traceback


# -------------------------------------------------------------------
# 1) Resolve ambiguity WITHOUT touching the index type
# -------------------------------------------------------------------
def _resolve_date_ambiguity(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the duplicate 'date' column if 'date' is also the index name.

    The index (DatetimeIndex) is preserved so downstream alignment in
    `add_exog_data` / `add_lag28` keeps working exactly as in the
    original direct_mlp.py / recursive_mlp.py scripts.
    """
    out = df.copy()
    if out.index.name == "date" and "date" in out.columns:
        out = out.drop(columns=["date"])
    return out


def _sorted_train(df: pd.DataFrame):
    """Sort chronologically, preserving the index type, and return dates."""
    if df.index.name == "date":
        df = df.sort_index()
        dates = pd.DatetimeIndex(df.index, name="date")
    elif "date" in df.columns:
        df = df.sort_values("date")          # RangeIndex / plain index
        dates = pd.DatetimeIndex(df["date"], name="date")
    else:
        raise KeyError("Neither a 'date' index nor a 'date' column was found.")
    return df, dates


# -------------------------------------------------------------------
# 2) Expanding-window folds — preserves train_df's index
# -------------------------------------------------------------------
def make_cv_folds(train_df: pd.DataFrame,
                  horizon: int = 28,
                  n_folds: int = 4,
                  gap: int = 0,
                  min_train_size: int = 100,
                  serie: str | None = None):
    df = _resolve_date_ambiguity(train_df)
    df, dates = _sorted_train(df)

    n = len(df)
    total_val = n_folds * horizon
    if n - total_val - gap < max(min_train_size, horizon + 30):
        raise ValueError("Insufficient history for the requested CV scheme.")

    y_sorted = df[serie].to_numpy() if serie is not None else None

    folds = []
    for k in range(n_folds):
        val_end   = n - (n_folds - 1 - k) * horizon
        val_start = val_end - horizon
        train_end = val_start - gap

        val_dates   = dates[val_start:val_end]
        fold_train  = df.iloc[:train_end].copy()          # index preserved
        y_true_fold = y_sorted[val_start:val_end] if y_sorted is not None else None

        folds.append((fold_train, val_dates, y_true_fold))
    return folds


# -------------------------------------------------------------------
# 3) Per-fold CV MASE — y_true is positional, no .loc[dates]
# -------------------------------------------------------------------
def _cv_mase_direct(params, folds, cov_df, serie, horizon=28, n_jobs=-1):
    mases = []
    for fold_train, val_dates, y_true in folds:
        stats = compute_train_stats(fold_train, cov_df)
        exog  = build_exog_features(cov_df, stats)

        X_tr, y_tr = build_train_base_direct(fold_train, exog, stats)
        X_va       = build_validation_base_direct(exog, y_tr, stats, val_dates)

        X_tr_e, y_tr_e = expand_train_horizons(X_tr, y_tr)
        X_va_h         = add_horizon_to_validation(X_va)

        model = DirectMLPForecaster(params=params,
                                    horizons=horizon, n_jobs=n_jobs)
        model.fit(X_tr_e, y_tr_e)
        y_hat = predict_model(X_va_h, stats, model, X_va_h.columns, True)

        m = compute_metrics(y_true, y_hat.to_numpy(),
                            fold_train[serie].to_numpy())
        mases.append(m["mase"])
    arr = np.asarray(mases, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.inf)
    return float(arr.mean()), arr.tolist()


def _cv_mase_recursive(params, folds, cov_df, serie):
    mases = []
    for fold_train, val_dates, y_true in folds:
        stats = compute_train_stats(fold_train, cov_df)
        exog  = build_exog_features(cov_df, stats)

        X_tr, y_tr = build_train_base_recursive(fold_train, exog, stats)
        X_va       = build_validation_base_recursive(exog, y_tr, stats, val_dates)

        model = RecursiveMLPForecaster(params=params)
        model.fit(X_tr, y_tr)
        y_hat = predict_model(X_va, stats, model, X_va.columns, True)

        m = compute_metrics(y_true, y_hat.to_numpy(),
                            fold_train[serie].to_numpy())
        mases.append(m["mase"])
    arr = np.asarray(mases, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.inf)
    return float(arr.mean()), arr.tolist()


# -------------------------------------------------------------------
# 4) Grid search — do NOT normalize train_df globally
# -------------------------------------------------------------------
def grid_search_cv(train_df, cov_df, serie,
                   horizon=28, n_folds=4, gap=0, n_jobs=-1,
                   param_grid=None) -> dict:
    if param_grid is None:
        param_grid = {
            "hidden_layer_sizes": [(32,), (64,)],
            "alpha":              [1e-4, 1e-3, 1e-2, 1e-1],
            "learning_rate_init": [1e-4, 1e-3, 1e-2],
        }

    keys   = list(param_grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]

    folds = make_cv_folds(train_df, horizon=horizon, n_folds=n_folds,
                          gap=gap, serie=serie)

    # Sanity print: does the first fold produce a non-empty training frame?
    ft0, vd0, _ = folds[0]
    print(f"fold 0: train rows={len(ft0)}  "
          f"val rows={len(vd0)}  "
          f"train index={type(ft0.index).__name__}")
    diag_stats = compute_train_stats(ft0, cov_df)
    diag_exog  = build_exog_features(cov_df, diag_stats)
    diag_X, diag_y = build_train_base_direct(ft0, diag_exog, diag_stats)
    print(f"fold 0 diagnostic: X_train rows={len(diag_X)}  "
          f"y_train rows={len(diag_y)}  "
          f"X_train index={type(diag_X.index).__name__}")
    if len(diag_X) == 0:
        raise RuntimeError(
            "build_train_base_direct returned 0 rows on fold 0. "
            "Check that exog_feats' index type matches fold_train's index."
        )

    direct_res, rec_res = [], []
    for i, hp in enumerate(combos, 1):
        hp_full = {**hp, "early_stopping": False}
        try:
            md, fmd = _cv_mase_direct(hp_full, folds, cov_df, serie,
                                      horizon, n_jobs)
        except Exception as e:
            print(f"[direct   ] FAILED {hp}: {e}")
            traceback.print_exc()
            md, fmd = float("inf"), []
        try:
            mr, fmr = _cv_mase_recursive(hp_full, folds, cov_df, serie)
        except Exception as e:
            print(f"[recursive] FAILED {hp}: {e}")
            traceback.print_exc()
            mr, fmr = float("inf"), []
        direct_res.append({"params": hp_full, "cv_mase": md, "fold_mases": fmd})
        rec_res.append   ({"params": hp_full, "cv_mase": mr, "fold_mases": fmr})
        print(f"[{i:02d}/{len(combos)}] {hp}  |  "
              f"direct MASE={md:.4f}   recursive MASE={mr:.4f}")

    return {
        "folds":             folds,
        "direct_results":    direct_res,
        "recursive_results": rec_res,
        "best_direct":       min(direct_res, key=lambda r: r["cv_mase"]),
        "best_recursive":    min(rec_res,    key=lambda r: r["cv_mase"]),
    }


# -------------------------------------------------------------------
# 5) Refit on the full training set + holdout evaluation
# -------------------------------------------------------------------
def refit_and_evaluate(train_df, val_df, cov_df, serie,
                       cv_out, val_dates,
                       horizon=28, n_jobs=-1):
    bd, br = cv_out["best_direct"], cv_out["best_recursive"]
    kind, chosen = (("direct", bd) if bd["cv_mase"] <= br["cv_mase"]
                    else ("recursive", br))
    print(f"\n==> Selected model: {kind.upper()}  "
          f"(best CV MASE = {chosen['cv_mase']:.4f})")
    print(f"    hyperparameters: {chosen['params']}")

    # train_df / val_df are used untouched — same contract as direct_mlp.py.
    stats = compute_train_stats(train_df, cov_df)
    exog  = build_exog_features(cov_df, stats)

    if kind == "direct":
        X_tr, y_tr     = build_train_base_direct(train_df, exog, stats)
        X_tr_e, y_tr_e = expand_train_horizons(X_tr, y_tr)
        X_va           = build_validation_base_direct(exog, y_tr, stats, val_dates)
        X_va_h         = add_horizon_to_validation(X_va)
        model = DirectMLPForecaster(params=chosen["params"],
                                    horizons=horizon, n_jobs=n_jobs)
        model.fit(X_tr_e, y_tr_e)
        y_hat = predict_model(X_va_h, stats, model, X_va_h.columns, True)
    else:
        X_tr, y_tr = build_train_base_recursive(train_df, exog, stats)
    X_va       = build_validation_base_recursive(exog, y_tr, stats, val_dates)
        model = RecursiveMLPForecaster(params=chosen["params"])
        model.fit(X_tr, y_tr)
        y_hat = predict_model(X_va, stats, model, X_va.columns, True)

    # Same lookup style as your original scripts.
    y_true  = val_df.set_index("date").loc[val_dates, serie].to_numpy()
    metrics = compute_metrics(y_true, y_hat.to_numpy(),
                              train_df[serie].to_numpy())
    print(f"    holdout metrics: {metrics}")
    return kind, chosen["params"], metrics

if __name__ == "__main__":
    train_df, val_df, cov_df = load_data()

    cv_out = grid_search_cv(train_df, cov_df, SERIE,
                            horizon=28, n_folds=4, gap=0, n_jobs=-1)

    refit_and_evaluate(train_df, val_df, cov_df, SERIE,
                       cv_out, VAL_DATES,
                       horizon=28, n_jobs=-1)