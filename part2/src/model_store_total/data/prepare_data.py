import pandas as pd
import numpy as np

from typing import Dict, List, Tuple

from ..config import (
    SERIE, PRECO_COL, ORIGEM, LAG, VAL_END, VAL_START
)
from .load_data import carregar_dados


# ---------- helpers ----------
def _ensure_datetime_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Garante DatetimeIndex sem mutar o DataFrame do chamador."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    if date_col in df.columns:
        return df.set_index(pd.to_datetime(df[date_col]))
    raise ValueError(f"DataFrame sem DatetimeIndex e sem coluna '{date_col}'")


def _get_dates(df: pd.DataFrame, date_col: str = "date") -> pd.Series:
    """Retorna as datas (do índice se já for DatetimeIndex, senão da coluna)."""
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=df.index)
    return pd.to_datetime(df[date_col])


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Colunas de features (exclui 'date' e o alvo cru)."""
    return [c for c in df.columns if c not in ("date", SERIE)]


# ---------- lags ----------
def add_lag28(df: pd.DataFrame, y_train: pd.Series, lag: int = LAG) -> pd.DataFrame:
    """Adiciona y(t-lag). y_train deve ter DatetimeIndex. Não muta df."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df precisa ter DatetimeIndex em add_lag28")
    df = df.copy()
    lag_dates = df.index - pd.Timedelta(days=lag) # type: ignore
    df[f"y_lag{lag}"] = y_train.reindex(lag_dates).to_numpy()
    return df


# ---------- estatísticas do treino ----------
def compute_train_stats(train_df: pd.DataFrame, cov_df: pd.DataFrame) -> Dict[str, float]:
    """Média e desvio EXCLUSIVAMENTE com dados do treino (sem leakage)."""
    y_mean = float(train_df[SERIE].mean())
    y_std = float(train_df[SERIE].std(ddof=0))

    dates = _get_dates(cov_df)
    cov_train = cov_df.loc[dates <= ORIGEM]

    preco_mean = float(cov_train[PRECO_COL].mean())
    preco_std = float(cov_train[PRECO_COL].std(ddof=0))

    return {"y_mean": y_mean, "y_std": y_std,
            "preco_mean": preco_mean, "preco_std": preco_std}


# ---------- features exógenas ----------
def build_exog_features(exog_df: pd.DataFrame, stats: Dict[str, float]) -> pd.DataFrame:
    """Constrói features exógenas para treino E validação.

    Retorna DataFrame com DatetimeIndex, coluna 'date' preservada (para merge)
    e colunas: dow_1..dow_6, snap_CA, is_event, event_*, preco_std.
    """
    exog_df = _ensure_datetime_index(exog_df)

    feats = pd.DataFrame(index=exog_df.index)
    feats["date"] = exog_df.index

    # 1.1 Dia da semana (segunda = referência)
    dow = pd.Series(exog_df.index.day_of_week, index=exog_df.index) # type: ignore
    dow_dum = pd.get_dummies(dow, prefix="dow", drop_first=True).astype(int)
    feats = feats.join(dow_dum)  # dow_1..dow_6

    # 1.2 SNAP
    feats["snap_CA"] = exog_df["snap_CA"].astype(int)

    # 1.3 Eventos (combinando os dois slots)
    t1 = exog_df["event_type_1"].fillna("")
    t2 = exog_df["event_type_2"].fillna("")
    feats["is_event"] = ((t1 != "") | (t2 != "")).astype(int)
    for et in sorted(set(t1[t1 != ""]) | set(t2[t2 != ""])):
        feats[f"event_{et}"] = ((t1 == et) | (t2 == et)).astype(int)

    # 1.4 Preço padronizado (stats do treino)
    feats["preco_std"] = (
        (exog_df[PRECO_COL].astype(float) - stats["preco_mean"]) / stats["preco_std"]
    )

    return feats


# ---------- base de treino ----------

def build_train_base(
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
    train_df = _ensure_datetime_index(train_df)
    exog_feats = _ensure_datetime_index(exog_feats)

    y_raw = train_df[SERIE].astype(float)
    y_std = (y_raw - stats["y_mean"]) / stats["y_std"]

    # Features nas datas do treino, mais lag-28 puxado de y_std
    X = exog_feats.reindex(y_raw.index).copy()
    X = add_lag28(X, y_std)

    # Remove as 28 primeiras linhas (sem lag); checa NaN remanescente
    X = X.dropna(subset=[f"y_lag{LAG}"])
    assert X.notna().all().all(), "NaN inesperado em X_train após dropna do lag"

    # Padroniza cada coluna de X (exógenas + y_lag28) usando stats globais
    for col in X.columns:
        if col == "date":
            continue
        m_key, s_key = f"{col}_mean", f"{col}_std"
        if m_key in stats and s_key in stats:
            m, s = stats[m_key], stats[s_key]
        else:
            m = float(X[col].mean())
            s = float(X[col].std(ddof=0)) or 1.0  # evita div. por zero
            stats[m_key], stats[s_key] = m, s
        X[col] = (X[col] - m) / s

    y = y_std.loc[X.index]

    return X.drop(columns="date"), y

# ---------- base de validação (novo) ----------
def build_validation_base(
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

    val_dates = pd.date_range(VAL_START, VAL_END, freq="D")
    X_val = exog_feats.reindex(val_dates).copy()
    X_val = add_lag28(X_val, y_train_std)

    assert (val_dates - pd.Timedelta(days=LAG) <= ORIGEM).all(), \
        "Vazamento: lag-28 da validação puxa dado futuro"

    # Padroniza usando estatísticas do treino
    for col in X_val.columns:
        if col == "date":
            continue
        m_key, s_key = f"{col}_mean", f"{col}_std"
        assert m_key in stats and s_key in stats, (
            f"Estatística de padronização ausente para '{col}' "
            f"({m_key!r}/{s_key!r}); rode construir_base_treino primeiro."
        )
        X_val[col] = (X_val[col] - stats[m_key]) / stats[s_key]

    na = X_val.isna().sum()
    assert na.sum() == 0, f"NaN em features de validação: {na[na > 0].to_dict()}"

    return X_val.drop(columns="date") if "date" in X_val.columns else X_val

def expand_train_horizons(
    X_train_df: pd.DataFrame,
    y_train_std: pd.Series,
    max_h: int = 28
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Takes the 1-step train dataframe and expands it to 28 steps.
    Keeps the target and calendar features anchored to the TARGET date,
    and shifts the historical lags forward by h-1 to simulate the origin constraint.
    """
    expanded_X = []
    expanded_y = []
    
    for h in range(1, max_h + 1):
        X_h = X_train_df.copy()
        X_h["h"] = h
        
        # Shift the lag features forward by (h - 1). 
        # If we are predicting date D at horizon h, the forecast was made at D - h + 1.
        # The lag available at that origin was the lag from D - h + 1 - 28.
        lag_cols = [c for c in X_h.columns if "lag" in c.lower()]
        for col in lag_cols:
            X_h[col] = X_h[col].shift(h - 1)
        
        # Target stays completely aligned with the calendar features (DOW, events)
        y_h = y_train_std.copy()
        
        # Drop rows where shifting pushed the lag out of bounds
        valid_mask = X_h[lag_cols[0]].notna()
        
        expanded_X.append(X_h.loc[valid_mask])
        expanded_y.append(y_h.loc[valid_mask])
        
    X_train_h = pd.concat(expanded_X).reset_index(drop=True)
    y_train_h = pd.concat(expanded_y).reset_index(drop=True)
    
    return X_train_h, y_train_h


def add_horizon_to_validation(X_validation):
    """
    Adds h=1..28 to the validation dataframe. 
    Assumes X_validation has exactly 28 rows in chronological order.
    """
    X_val_h = X_validation.copy()
    
    # FIX 2: Freeze sales lags at the origin (first row = 2016-03-27)
    # Identify your lag columns. If they are all like 'y_lag_1', 'y_lag_28', etc.
    lag_cols = [c for c in X_val_h.columns if "lag" in c.lower() or "roll" in c.lower()]
    
    if lag_cols:
        origin_lags = X_val_h.iloc[0][lag_cols].copy()
        X_val_h[lag_cols] = origin_lags.values  # Broadcast to all 28 rows
        
    # Now assign the true horizons 1 to 28
    X_val_h["h"] = np.arange(1, 29)
    return X_val_h

def get_all_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray]:
    """
    Runs the whole data pipeline, returning the final expanded train dataframe
    and series and the validation dataframe with the horizon column.
    
    Returns:
        pd.DataFrame: X_train_expanded_df
        pd.Series: y_train_expanded
        pd.DataFrame: X_validation_horizon
        np.ndarray: y_true
    """
    train_df, val_df, cov_df = carregar_dados()
            
    # 1) Estatísticas do treino (antes de construir features)
    stats = compute_train_stats(train_df, cov_df)

    # 2) Features exógenas sobre treino + validação (uma única vez)
    exog_feats = build_exog_features(cov_df, stats)

    # 3) Bases de treino e validação
    X_train_df, y_train_std = build_train_base(train_df, exog_feats, stats)
    X_validation = build_validation_base(exog_feats, y_train_std, stats)

    # 4) Bases expandidas para os 28 h's
    X_train_expanded_df, y_train_expanded = expand_train_horizons(X_train_df, y_train_std)
    X_validation_horizon = add_horizon_to_validation(X_validation)
    
    val_dates = pd.date_range(VAL_START, VAL_END, freq="D")
    y_true = val_df.set_index("date").loc[val_dates, SERIE].to_numpy()
    
    return (
        X_train_expanded_df,
        y_train_expanded,
        X_validation_horizon,
        y_true,
    )
    

if __name__ == "__main__":
    train_df, val_df, cov_df = carregar_dados()
        
    # 1) Estatísticas do treino (antes de construir features)
    stats = compute_train_stats(train_df, cov_df)

    # 2) Features exógenas sobre treino + validação (uma única vez)
    exog_feats = build_exog_features(cov_df, stats)

    # 3) Bases de treino e validação
    X_train_df, y_train_std = build_train_base(train_df, exog_feats, stats)
    X_validation = build_validation_base(exog_feats, y_train_std, stats)
    
    # 4) Bases expandidas para os 28 h's
    X_train_expanded_df, y_train_expanded = expand_train_horizons(X_train_df, y_train_std)
    X_validation_horizon = add_horizon_to_validation(X_validation)
    
    
    print("Train data")
    print(X_train_df.head())
    print(f"Num of rows: {len(X_train_df)}")
    print("Expanded train data")
    print(X_train_expanded_df.head())
    print(f"Num of rows: {len(X_train_expanded_df)}")
    print(50 * "-")
    
    print("Validation data")
    print(X_validation.head())
    print("Validation data with horizon")
    print(X_validation_horizon.head())