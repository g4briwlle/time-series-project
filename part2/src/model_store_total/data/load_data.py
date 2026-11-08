import pandas as pd

from typing import Tuple

from ..config import DATA_DIR

def _load(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df.sort_index()
    df = df.reset_index()
    df = df.set_index(pd.to_datetime(df["date"]))
    return df

def carregar_dados() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega os dados brutos e garante ordenação temporal."""
    return (
        _load(DATA_DIR / "treino.csv"),
        _load(DATA_DIR / "validacao.csv"),
        _load(DATA_DIR / "covariaveis.csv"),
    )
    