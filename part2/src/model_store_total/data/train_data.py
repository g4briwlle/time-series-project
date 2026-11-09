import pandas as pd

from typing import Dict, Tuple

from ..config import (
    SERIE, LAG, LAGS_RECURSIVE
)
from .data_utils import _ensure_datetime_index, add_lag28


# ---------- base de treino ----------
def add_exog_data(
    train_df: pd.DataFrame,
    exog_feats: pd.DataFrame,
):
    """
    Joins train dataframe with exogenous features, without any lags or rolling windows.
    """
    
    train_df = _ensure_datetime_index(train_df)
    exog_feats = _ensure_datetime_index(exog_feats)

    y_raw = train_df[SERIE].astype(float)

    # Features nas datas do treino, mais lag-28 puxado de y_std
    X = exog_feats.reindex(y_raw.index).copy()
    
    return X

def standardize_train(X, y_std, stats):
    # Padroniza cada coluna de X, pulando as que já foram baseadas em y_std
    for col in X.columns:
        if col == "date" or "lag" in col or "roll" in col:
            continue
            
        m_key, s_key = f"{col}_mean", f"{col}_std"
        if m_key in stats and s_key in stats:
            m, s = stats[m_key], stats[s_key]
        else:
            m = float(X[col].mean())
            s = float(X[col].std(ddof=0)) or 1.0  
            stats[m_key], stats[s_key] = m, s
        X[col] = (X[col] - m) / s

    y = y_std.loc[X.index]
    return X.drop(columns="date") if "date" in X.columns else X, y

def build_train_base_direct(
    train_df: pd.DataFrame,
    exog_feats: pd.DataFrame,
    stats: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Monta X_train e y_train padronizado, alinhados por data.

    Retorna (X_train, y_train_std):
        - X_train: features padronizadas (já com y_lag28, sem 'date')
        - y_train_std: alvo padronizado (mesmas datas de X_train)

    Efeito colateral: preenche `stats` com `<col>_mean`/`<col>_std` para cada
    coluna de X (exógenas + y_lag28), reutilizando valores já presentes para
    permitir consistência entre treino e inferência.
    """

    y_raw = train_df[SERIE].astype(float)
    y_std = (y_raw - stats["y_mean"]) / stats["y_std"]

    X = add_exog_data(train_df, exog_feats)
    X = add_lag28(X, y_std)

    # Remove as 28 primeiras linhas (sem lag); checa NaN remanescente
    X = X.dropna(subset=[f"y_lag{LAG}"])
    assert X.notna().all().all(), "NaN inesperado em X_train após dropna do lag"

    return standardize_train(X, y_std, stats)

def build_train_base_recursive(
    train_df: pd.DataFrame,
    exog_feats: pd.DataFrame,
    stats: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.Series]:
    y_raw = train_df[SERIE].astype(float)
    y_std = (y_raw - stats["y_mean"]) / stats["y_std"]

    X = add_exog_data(train_df, exog_feats)
    
    # Selected recent and seasonal lags
    for lag in LAGS_RECURSIVE:
        X[f'lag_{lag}'] = y_std.shift(lag)
        
    # Rolling window statistics (computed on lag_1 to avoid looking at current target)
    X['rolling_mean_7'] = X['lag_1'].rolling(window=7).mean()
    X['rolling_mean_28'] = X['lag_1'].rolling(window=28).mean()
    X['rolling_std_7'] = X['lag_1'].rolling(window=7).std()
    
    feature_cols = [f'lag_{lag}' for lag in LAGS_RECURSIVE] + ['rolling_mean_7', 'rolling_mean_28', 'rolling_std_7']
    X = X.dropna(subset=feature_cols)
    
    return standardize_train(X, y_std, stats)