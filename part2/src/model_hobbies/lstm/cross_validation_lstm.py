from __future__ import annotations

import itertools
import traceback
import numpy as np
import pandas as pd

from ..data.load_data import load_data
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
from ...model_store_total.models.models_utils import predict_model, compute_metrics, save_outputs
from ..config import SERIE, VAL_DATES, OUTPUT_DIR


from .direct_lstm_class import DirectLSTMForecaster
from .recursive_lstm_class import RecursiveLSTMForecaster

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


def _cv_mase_direct_lstm(params: dict, folds: list, cov_df: pd.DataFrame, serie: str, horizon: int = 28) -> tuple[float, list[float]]:
    mases = []
    for fold_train, val_dates, y_true in folds:
        stats = compute_train_stats(fold_train, cov_df)
        exog = build_exog_features(cov_df, stats)

        X_tr, y_tr = build_train_base_direct(fold_train, exog, stats)
        X_va = build_validation_base_direct(exog, y_tr, stats, val_dates)

        X_tr_e, y_tr_e = expand_train_horizons(X_tr, y_tr)
        X_va_h = add_horizon_to_validation(X_va)

        model = DirectLSTMForecaster(params=params, horizons=horizon)
        model.fit(X_tr_e, y_tr_e)
        y_hat = predict_model(X_va_h, stats, model, X_va_h.columns, True)

        m = compute_metrics(y_true, y_hat.to_numpy(), fold_train[serie].to_numpy())
        mases.append(m["mase"])

    arr = np.asarray(mases, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.inf)
    return float(arr.mean()), arr.tolist()


def _cv_mase_recursive_lstm(params: dict, folds: list, cov_df: pd.DataFrame, serie: str) -> tuple[float, list[float]]:
    mases = []
    for fold_train, val_dates, y_true in folds:
        stats = compute_train_stats(fold_train, cov_df)
        exog = build_exog_features(cov_df, stats)

        X_tr, y_tr = build_train_base_recursive(fold_train, exog, stats)
        X_va = build_validation_base_recursive(exog, y_tr, stats, val_dates)

        model = RecursiveLSTMForecaster(params=params)
        model.fit(X_tr, y_tr)
        y_hat = predict_model(X_va, stats, model, X_va.columns, True)

        m = compute_metrics(y_true, y_hat.to_numpy(), fold_train[serie].to_numpy())
        mases.append(m["mase"])

    arr = np.asarray(mases, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.inf)
    return float(arr.mean()), arr.tolist()


def grid_search_cv_lstm(
    train_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    serie: str,
    horizon: int = 28,
    n_folds: int = 4,
    gap: int = 0,
    param_grid: dict | None = None,
) -> dict:
    if param_grid is None:
        param_grid = {
            "hidden_size": [16, 32],
            "seq_len": [7, 14],
            "lr": [1e-3, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "epochs": [35],
            "mode": ["pooled"],  # troque para ["pooled", "independent"] se quiser testar ambos
        }

    keys = list(param_grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]
    folds = make_cv_folds(train_df, horizon=horizon, n_folds=n_folds, gap=gap, serie=serie)

    direct_res, rec_res = [], []
    for i, hp in enumerate(combos, 1):
        try:
            md, fmd = _cv_mase_direct_lstm(hp, folds, cov_df, serie, horizon)
        except Exception as e:
            print(f"[LSTM direct   ] FAILED {hp}: {e}")
            traceback.print_exc()
            md, fmd = float("inf"), []

        try:
            mr, fmr = _cv_mase_recursive_lstm(hp, folds, cov_df, serie)
        except Exception as e:
            print(f"[LSTM recursive] FAILED {hp}: {e}")
            traceback.print_exc()
            mr, fmr = float("inf"), []

        direct_res.append({"params": hp, "cv_mase": md, "fold_mases": fmd})
        rec_res.append({"params": hp, "cv_mase": mr, "fold_mases": fmr})
        print(f"[{serie} | {i:02d}/{len(combos)}] {hp} -> Direct MASE={md:.4f} | Recursive MASE={mr:.4f}")

    return {
        "folds": folds,
        "direct_results": direct_res,
        "recursive_results": rec_res,
        "best_direct": min(direct_res, key=lambda r: r["cv_mase"]),
        "best_recursive": min(rec_res, key=lambda r: r["cv_mase"]),
    }


def refit_and_evaluate_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    serie: str,
    cv_out: dict,
    val_dates: pd.DatetimeIndex,
    horizon: int = 28,
    save: bool = False,
):
    bd, br = cv_out["best_direct"], cv_out["best_recursive"]
    kind, chosen = ("direct", bd) if bd["cv_mase"] <= br["cv_mase"] else ("recursive", br)

    print(f"\n==> [{serie}] Melhor Estratégia LSTM (via CV no treino): {kind.upper()} (CV MASE = {chosen['cv_mase']:.4f})")
    print(f"    Hiperparâmetros escolhidos: {chosen['params']}")

    stats = compute_train_stats(train_df, cov_df)
    exog = build_exog_features(cov_df, stats)

    if kind == "direct":
        X_tr, y_tr = build_train_base_direct(train_df, exog, stats)
        X_tr_e, y_tr_e = expand_train_horizons(X_tr, y_tr)
        X_va = build_validation_base_direct(exog, y_tr, stats, val_dates)
        X_va_h = add_horizon_to_validation(X_va)

        model = DirectLSTMForecaster(params=chosen["params"], horizons=horizon)
        model.fit(X_tr_e, y_tr_e)
        y_hat = predict_model(X_va_h, stats, model, X_va_h.columns, True)
    else:
        X_tr, y_tr = build_train_base_recursive(train_df, exog, stats)
        X_va = build_validation_base_recursive(exog, y_tr, stats, val_dates)

        model = RecursiveLSTMForecaster(params=chosen["params"])
        model.fit(X_tr, y_tr)
        y_hat = predict_model(X_va, stats, model, X_va.columns, True)

    y_true = val_df.set_index("date").loc[val_dates, serie].to_numpy()
    metrics = compute_metrics(y_true, y_hat.to_numpy(), train_df[serie].to_numpy())
    print(f"    Métricas na Validação (28 dias): {metrics}")

    if save:
        out_dir = OUTPUT_DIR.parent.parent / "part4" / "outputs"
        # Nota: o enunciado aceita 'rede' ou 'lstm' na coluna modelo
        save_outputs(y_hat, val_dates, metrics, model_name="rede", output_dir=out_dir)

    return kind, chosen["params"], metrics, y_hat


if __name__ == "__main__":
    train_df, val_df, cov_df = load_data()
    cv_out = grid_search_cv_lstm(train_df, cov_df, SERIE, horizon=28, n_folds=4)
    refit_and_evaluate_lstm(train_df, val_df, cov_df, SERIE, cv_out, VAL_DATES, horizon=28, save=True)