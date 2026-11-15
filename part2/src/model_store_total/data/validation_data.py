import pandas as pd

from typing import Dict, Tuple

from ..config import (
    ORIGEM, VAL_DATES, LAGS_RECURSIVE, LAG
)
from .data_utils import _ensure_datetime_index, _add_recursive_features, add_lag28


def standardize_validation(X_val, stats):
    # Only standardize exogenous columns; dynamically added lags will already be standardized
    for col in X_val.columns:
        if col == "date" or "lag" in col or "roll" in col:
            continue
        m_key, s_key = f"{col}_mean", f"{col}_std"
        assert m_key in stats and s_key in stats, (
            f"Estatística de padronização ausente para '{col}' "
        )
        X_val[col] = (X_val[col] - stats[m_key]) / stats[s_key]

    na = X_val.isna().sum()

    return X_val.drop(columns="date") if "date" in X_val.columns else X_val

# ---------- base de validação (novo) ----------
def build_validation_base_direct(
    exog_feats: pd.DataFrame,
    y_train_std: pd.Series,
    stats: Dict[str, float],
    val_dates = VAL_DATES
) -> pd.DataFrame:
    """Monta X_validation: features exógenas no horizonte + lag-28 do treino,
    padronizadas com as estatísticas do treino (`stats`).

    Guarda-corpos:
        - t-28 <= ORIGEM para todo t no horizonte (sem vazamento);
        - sem NaN em nenhuma coluna após padronização.

    Importante: usa as médias/desvios gravados em `stats` por
    `construir_base_treino` — nunca recalcula em cima da validação (evita
    leakage de distribuição).
    """
    exog_feats = _ensure_datetime_index(exog_feats)
    
    X_val = exog_feats.reindex(val_dates).copy()
    X_val = add_lag28(X_val, y_train_std)

    assert (val_dates - pd.Timedelta(days=LAG) <= ORIGEM).all(), \
        "Vazamento: lag-28 da validação puxa dado futuro"

    return standardize_validation(X_val, stats)

def build_validation_base_recursive(
    exog_feats: pd.DataFrame,
    y_train_std: pd.Series,
    stats: Dict[str, float],
    val_dates=VAL_DATES,
) -> pd.DataFrame:
    """Recursive validation base: exog for all val dates + recursive features
    seeded from the training target history.

    Only the first row(s) whose lag/rolling references fall entirely inside
    y_train_std will be fully populated. Later rows intentionally contain NaN
    for the recursive features — the model's `.predict` fills them as it walks
    forward step by step.
    """
    exog_feats = _ensure_datetime_index(exog_feats)
    y_train_std = y_train_std.sort_index()

    print("y_train_std")
    print(y_train_std)
    print("val_dates")
    print(val_dates)

    X_val = exog_feats.reindex(val_dates).copy()
    X_val = _add_recursive_features(X_val, y_train_std)

    feature_cols = (
        [f"lag_{lag}" for lag in LAGS_RECURSIVE]
        + ["rolling_mean_7", "rolling_mean_28", "rolling_std_7"]
    )

    # The first validation row must be fully populated: every lag/rolling
    # reference it needs lives at or before ORIGEM, hence inside y_train_std.
    assert X_val.iloc[0][feature_cols].notna().all(), (
        "First validation row is missing recursive features — "
        "y_train_std does not cover the required history"
    )

    # No leakage of future target into the seed row.
    assert (val_dates[0] - pd.Timedelta(days=1)) <= ORIGEM, \
        "Vazamento: seed da validação recursiva usa dado futuro"

    return standardize_validation(X_val, stats)