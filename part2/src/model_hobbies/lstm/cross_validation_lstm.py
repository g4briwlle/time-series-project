"""
cross_validation_lstm.py

Complete file — replaces the previous version.

Changed vs. previous version:
  - grid_search_cv_lstm now hoists per-fold data prep out of the hp loop
    (4 prep calls instead of 64) and lazily caches sliding windows per
    (fold, seq_len, mode) with eviction on config change.
  - _cv_mase_direct_lstm / _cv_mase_recursive_lstm: REMOVED. Replaced by
    _cv_direct_cached / _cv_recursive_cached.
  - New helpers: _prepare_direct_fold, _prepare_recursive_fold,
    _recursive_fit_one.
  - refit_and_evaluate_lstm: unchanged in behaviour; RECONSTRUCTED here —
    verify against your original.
"""

import itertools
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..data.prepare_data import (
    compute_train_stats,
    build_exog_features,
    build_train_base_direct,
    build_validation_base_direct,
    build_train_base_recursive,
    build_validation_base_recursive,
    expand_train_horizons,
    add_horizon_to_validation,
)
from .model_utils import predict_model, compute_metrics
from .direct_lstm_class import DirectLSTMForecaster
from .recursive_lstm_class import RecursiveLSTMForecaster
# ─────────────────────────────────────────────────────────────────────


# ═════════════════════════════════════════════════════════════════════
#  Per-fold preparation (params-independent — called once per fold)
# ═════════════════════════════════════════════════════════════════════



def _prepare_direct_fold(fold_train, val_dates, y_true, cov_df, serie):
    """Everything the direct model needs from a fold, computed once."""
    stats = compute_train_stats(fold_train, cov_df)
    exog  = build_exog_features(cov_df, stats)
    X_tr, y_tr     = build_train_base_direct(fold_train, exog, stats)
    X_va           = build_validation_base_direct(exog, y_tr, stats, val_dates)
    X_tr_e, y_tr_e = expand_train_horizons(X_tr, y_tr)
    X_va_h         = add_horizon_to_validation(X_va)
    return dict(
        stats=stats,
        X_tr_e=X_tr_e,
        y_tr_e=y_tr_e,
        X_va_h=X_va_h,
        y_true=y_true,
        train_serie=fold_train[serie].to_numpy(),
    )


def _prepare_recursive_fold(fold_train, val_dates, y_true, cov_df, serie):
    """Everything the recursive model needs from a fold, computed once."""
    stats = compute_train_stats(fold_train, cov_df)
    exog  = build_exog_features(cov_df, stats)
    X_tr, y_tr = build_train_base_recursive(fold_train, exog, stats)
    X_va       = build_validation_base_recursive(exog, y_tr, stats, val_dates)
    return dict(
        stats=stats,
        X_tr=X_tr,
        y_tr=y_tr,
        X_va=X_va,
        y_true=y_true,
        train_serie=fold_train[serie].to_numpy(),
    )


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


# ═════════════════════════════════════════════════════════════════════
#  Per-combo CV evaluation using cached artefacts
# ═════════════════════════════════════════════════════════════════════

def _cv_direct_cached(hp, folds, windows_cache, horizon):
    """
    windows_cache: {fold_idx: precomputed_windows_dict}
    (already for the correct seq_len / mode of `hp`).
    """
    mases = []
    for fi, payload in enumerate(folds):
        try:
            model = DirectLSTMForecaster(
                params=hp, horizons=horizon, device="cuda",
            )
            model.fit(
                payload["X_tr_e"],
                payload["y_tr_e"],
                precomputed_windows=windows_cache[fi],
            )
            y_hat = predict_model(
                payload["X_va_h"], payload["stats"],
                model, payload["X_va_h"].columns, True,
            )
            m = compute_metrics(
                payload["y_true"], y_hat.to_numpy(), payload["train_serie"],
            )
            mases.append(m["mase"])
        except Exception as e:
            print(f"[direct fold {fi}] FAILED: {e}")
            mases.append(np.inf)

    arr = np.asarray(mases, dtype=float)
    return float(np.where(np.isfinite(arr), arr, np.inf).mean())


def _recursive_fit_one(payload, hp):
    """Module-level so joblib/loky can pickle it."""
    model = RecursiveLSTMForecaster(params=hp)
    model.fit(payload["X_tr"], payload["y_tr"])
    try:
        y_hat = predict_model(
            payload["X_va"], payload["stats"],
            model, payload["X_va"].columns, True,
        )
        m = compute_metrics(
            payload["y_true"], y_hat.to_numpy(), payload["train_serie"],
        )
        return m["mase"]
    except Exception as e:
        print(f"[recursive fold] FAILED: {e}")
        return np.inf


def _cv_recursive_cached(hp, folds):
    mases = Parallel(n_jobs=-1)(
        delayed(_recursive_fit_one)(p, hp) for p in folds
    )
    arr = np.asarray(mases, dtype=float)
    return float(np.where(np.isfinite(arr), arr, np.inf).mean())


# ═════════════════════════════════════════════════════════════════════
#  Public entry point
# ═════════════════════════════════════════════════════════════════════

def grid_search_cv_lstm(
    train_df, cov_df, serie, horizon=28, n_folds=4, gap=0, param_grid=None,
):
    if param_grid is None:
        param_grid = {
            "hidden_size":  [16, 32],
            "seq_len":      [7, 14],
            "lr":           [1e-3, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "epochs":       [35],
            "mode":         ["pooled"],
        }

    keys   = list(param_grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]
    folds  = make_cv_folds(
        train_df, horizon=horizon, n_folds=n_folds, gap=gap, serie=serie,
    )

    # ── 1. prep once per fold (params-independent) ────────────────
    direct_folds = [
        _prepare_direct_fold(ft, vd, yt, cov_df, serie)
        for ft, vd, yt in folds
    ]
    rec_folds = [
        _prepare_recursive_fold(ft, vd, yt, cov_df, serie)
        for ft, vd, yt in folds
    ]

    # ── 2. helper instance — only used for _prepare_windows ───────
    #    _prepare_windows needs self.h_col and (via _encode_h) self.horizons.
    helper = DirectLSTMForecaster(params={}, horizons=horizon)

    # ── 3. hp loop with LAZY per-config window cache ──────────────
    #    combos from itertools.product are ordered so that consecutive
    #    entries share (seq_len, mode) — the cache only rebuilds on
    #    config change, keeping at most ONE config's windows alive.
    current_config = None
    windows_cache: dict[int, dict] = {}
    direct_res, rec_res = [], []

    for i, hp in enumerate(combos, 1):
        seq_len = hp.get("seq_len", 7)
        mode    = hp.get("mode", "pooled")
        config  = (seq_len, mode)

        if config != current_config:
            windows_cache.clear()
            for fi, payload in enumerate(direct_folds):
                windows_cache[fi] = helper._prepare_windows(
                    payload["X_tr_e"], payload["y_tr_e"], seq_len, mode,
                )
            current_config = config

        md = _cv_direct_cached(hp, direct_folds, windows_cache, horizon)
        mr = _cv_recursive_cached(hp, rec_folds)

        direct_res.append({"params": hp, "cv_mase": md})
        rec_res.append({"params": hp, "cv_mase": mr})
        print(
            f"[{serie} | {i:02d}/{len(combos)}] {hp} -> "
            f"Direct MASE={md:.4f} | Recursive MASE={mr:.4f}"
        )

    return {
        "folds": folds,
        "direct_results": direct_res,
        "recursive_results": rec_res,
        "best_direct": min(direct_res, key=lambda r: r["cv_mase"]),
        "best_recursive": min(rec_res, key=lambda r: r["cv_mase"]),
    }


# ═════════════════════════════════════════════════════════════════════
#  Refit + final validation — UNCHANGED BEHAVIOUR, reconstructed
#  ⚠ verify signature/body against your original
# ═════════════════════════════════════════════════════════════════════

def refit_and_evaluate_lstm(
    full_train, val_dates, y_true, cov_df, serie,
    best_params, model_type="direct", horizon=28,
):
    """
    Refit the best params on the full training set and score on validation.
    Only does ONE fit, so it doesn't need the window cache.
    """
    stats = compute_train_stats(full_train, cov_df)
    exog  = build_exog_features(cov_df, stats)

    if model_type == "direct":
        X_tr, y_tr = build_train_base_direct(full_train, exog, stats)
        X_va       = build_validation_base_direct(exog, y_tr, stats, val_dates)
        model = DirectLSTMForecaster(
            params=best_params, horizons=horizon, device="cuda",
        )
        model.fit(X_tr, y_tr)
        X_eval = add_horizon_to_validation(X_va)
    elif model_type == "recursive":
        X_tr, y_tr = build_train_base_recursive(full_train, exog, stats)
        X_va       = build_validation_base_recursive(exog, y_tr, stats, val_dates)
        model = RecursiveLSTMForecaster(params=best_params)
        model.fit(X_tr, y_tr)
        X_eval = X_va
    else:
        raise ValueError(f"model_type inválido: {model_type}")

    y_hat = predict_model(X_eval, stats, model, X_eval.columns, True)
    m     = compute_metrics(y_true, y_hat.to_numpy(), full_train[serie].to_numpy())
    return model, y_hat, m