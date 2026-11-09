import pandas as pd
import numpy as np

from typing import Dict, List

from ..config import (
    SERIE, PRECO_COL, ORIGEM, LAG
)


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