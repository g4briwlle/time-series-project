import pandas as pd
from typing import Dict, List, Tuple

from .config_store_total import (
    SERIE, PRECO_COL, ORIGEM, LAG, VAL_END, VAL_START
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
    lag_dates = df.index - pd.Timedelta(days=lag)
    df[f"y_lag{lag}"] = y_train.reindex(lag_dates).to_numpy()
    return df


# ---------- estatísticas do treino ----------
def calcular_estatisticas_treino(train_df: pd.DataFrame, cov_df: pd.DataFrame) -> Dict[str, float]:
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
    dow = pd.Series(exog_df.index.day_of_week, index=exog_df.index)
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

def construir_base_treino(
    train_df: pd.DataFrame,
    exog_feats: pd.DataFrame,
    stats: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Monta X_train e y_train padronizado, alinhados por data.

    Retorna (X_train, y_train_std):
        - X_train: features (já com y_lag28, sem 'date')
        - y_train_std: alvo padronizado (mesmas datas de X_train)
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

    y = y_std.loc[X.index]

    return X.drop(columns="date"), y


# ---------- base de validação (novo) ----------

def construir_base_validacao(
    exog_feats: pd.DataFrame,
    y_train_std: pd.Series,
) -> pd.DataFrame:
    """Monta X_validation: features exógenas no horizonte + lag-28 do treino.

    Guarda-corpos:
        - t-28 <= ORIGEM para todo t no horizonte (sem vazamento);
        - sem NaN em nenhuma coluna.
    """
    exog_feats = _ensure_datetime_index(exog_feats)

    val_dates = pd.date_range(VAL_START, VAL_END, freq="D")
    X_val = exog_feats.reindex(val_dates).copy()
    X_val = add_lag28(X_val, y_train_std)

    assert (val_dates - pd.Timedelta(days=LAG) <= ORIGEM).all(), \
        "Vazamento: lag-28 da validação puxa dado futuro"

    na = X_val.isna().sum()
    assert na.sum() == 0, f"NaN em features de validação: {na[na > 0].to_dict()}"

    return X_val.drop(columns="date") if "date" in X_val.columns else X_val
