import pandas as pd

from typing import Dict

from ..config import (
    ORIGEM, LAG, VAL_DATES, LAGS_RECURSIVE
)
from .load_data import carregar_dados
from .data_utils import _ensure_datetime_index, add_lag28


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
    assert na.sum() == 0, f"NaN em features de validação: {na[na > 0].to_dict()}"

    return X_val.drop(columns="date") if "date" in X_val.columns else X_val

# ---------- base de validação (novo) ----------
def build_validation_base_direct(
    exog_feats: pd.DataFrame,
    y_train_std: pd.Series,
    stats: Dict[str, float],
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

    X_val = exog_feats.reindex(VAL_DATES).copy()
    X_val = add_lag28(X_val, y_train_std)

    assert (VAL_DATES - pd.Timedelta(days=LAG) <= ORIGEM).all(), \
        "Vazamento: lag-28 da validação puxa dado futuro"

    return standardize_validation(X_val, stats)
    

def build_validation_base_recursive(
    exog_feats: pd.DataFrame,
    stats: Dict[str, float],
) -> pd.DataFrame:
    exog_feats = _ensure_datetime_index(exog_feats)
    
    X_val = exog_feats.reindex(VAL_DATES).copy()
    
    return standardize_validation(X_val, stats)
    