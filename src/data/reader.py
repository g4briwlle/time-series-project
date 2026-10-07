# src/data/reader.py
from enum import StrEnum
from pathlib import Path
from typing import Literal

import pandas as pd

from ._config import (
    TRAIN_PATH,
    VALIDATION_PATH,
    HOLDOUT_PATH,
    DATE_COLUMN,
    TRAIN_START, TRAIN_END,
    VALIDATION_START, VALIDATION_END,
    HOLDOUT_START, HOLDOUT_END,
)

SplitName = Literal["train", "validation", "holdout"]


class SeriesNames(StrEnum):
    STORE_TOTAL = "store_total"
    FOODS = "FOODS"
    HOBBIES = "HOBBIES"


SERIES: list[str] = [s.value for s in SeriesNames]

_PATHS: dict[str, Path] = {
    "train": TRAIN_PATH,
    "validation": VALIDATION_PATH,
    "holdout": HOLDOUT_PATH,
}

_BOUNDS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train": (pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)),
    "validation": (pd.Timestamp(VALIDATION_START), pd.Timestamp(VALIDATION_END)),
    "holdout": (pd.Timestamp(HOLDOUT_START), pd.Timestamp(HOLDOUT_END)),
}


def _validate_series_name(series_name: str) -> None:
    if series_name not in SERIES:
        raise ValueError(
            f"Série inválida: {series_name!r}. Use uma de {SERIES}."
        )


def _read_csv(split_category: SplitName) -> pd.DataFrame:
    if split_category not in _PATHS:
        raise ValueError(
            f"Split inválido: {split_category!r}. "
            f"Use um de {list(_PATHS)}."
        )

    path = _PATHS[split_category]
    df = pd.read_csv(path, parse_dates=[DATE_COLUMN])

    if DATE_COLUMN not in df.columns:
        raise ValueError(f"{path} não tem coluna {DATE_COLUMN!r}.")

    return df.sort_values(DATE_COLUMN).reset_index(drop=True)


def _validate_split(df: pd.DataFrame, split_category: SplitName) -> None:
    """Garante invariantes temporais de um único split. Roda no __init__."""
    start, end = _BOUNDS[split_category]

    # 1. Holdout não pode ter y
    if split_category == "holdout":
        vazadas = [c for c in SERIES if c in df.columns]
        if vazadas:
            raise ValueError(
                f"holdout_datas.csv não pode ter colunas de y: {vazadas}."
            )
    else:
        faltando = [c for c in SERIES if c not in df.columns]
        if faltando:
            raise ValueError(
                f"{split_category} sem colunas de série: {faltando}."
            )

    # 2. Datas dentro dos limites do split
    datas = pd.DatetimeIndex(df[DATE_COLUMN].unique())
    if datas.min() < start or datas.max() > end:
        raise ValueError(
            f"{split_category} fora dos limites "
            f"[{start.date()}, {end.date()}]: "
            f"encontrado [{datas.min().date()}, {datas.max().date()}]."
        )

    # 3. Sem datas duplicadas
    if df[DATE_COLUMN].duplicated().any():
        dup = df.loc[df[DATE_COLUMN].duplicated(), DATE_COLUMN].dt.date.tolist()
        raise ValueError(f"{split_category} tem datas duplicadas: {dup[:5]}...")

    # 4. Frequência diária contínua
    esperado = pd.date_range(datas.min(), datas.max(), freq="D")
    if not datas.sort_values().equals(esperado):
        raise ValueError(
            f"{split_category} não é diário contínuo "
            f"({len(datas)} linhas, esperado {len(esperado)})."
        )

    # 5. Validação tem exatamente 28 dias (horizonte da Task 1)
    if split_category == "validation" and len(datas) != 28:
        raise ValueError(
            f"validação deve ter 28 dias, tem {len(datas)}."
        )


class DataReader:
    """
    Ponto único de acesso aos splits temporais.

    Garante no __init__ que o split é internamente consistente
    (limites, sem duplicatas, frequência diária, colunas corretas).
    Use `validate_all_splits()` para checar também a ausência de
    sobreposição entre treino, validação e holdout.
    """

    def __init__(self, split_category: SplitName):
        if split_category not in _PATHS:
            raise ValueError(
                f"Split inválido: {split_category!r}. "
                f"Use um de {list(_PATHS)}."
            )
        self.split_category: SplitName = split_category
        self.df: pd.DataFrame = _read_csv(split_category)
        _validate_split(self.df, split_category)

    # ---------- Formatos de saída ----------

    def to_long(self) -> pd.DataFrame:
        """
        Retorna em formato longo: colunas 'date', 'series', 'y'.
        Disponível apenas em treino e validação.
        """
        if self.split_category == "holdout":
            raise ValueError(
                "holdout não tem y; use `get_dates()` para obter as datas."
            )
        return (
            self.df.melt(
                id_vars=DATE_COLUMN,
                value_vars=SERIES,
                var_name="series",
                value_name="y",
            )
            .sort_values([DATE_COLUMN, "series"])
            .reset_index(drop=True)
        )

    def get_series(self, series_name: str) -> pd.Series:
        """
        Retorna uma série com DatetimeIndex diário, ordenada, nomeada.
        Disponível apenas em treino e validação.
        """
        _validate_series_name(series_name)
        if self.split_category == "holdout":
            raise ValueError("holdout não tem y; use `get_dates()`.")
        s = (
            self.df.set_index(DATE_COLUMN)[series_name]
            .sort_index()
            .asfreq("D")
        )
        s.name = series_name
        return s

    def get_dates(self) -> pd.DatetimeIndex:
        """Retorna apenas as datas, ordenadas e únicas."""
        return pd.DatetimeIndex(
            self.df[DATE_COLUMN].sort_values().unique()
        )

    # ---------- Utilitários de modelagem ----------

    def mase_scale(self, series_name: str, season: int = 7) -> float:
        """
        MAE in-sample do naive sazonal semanal, calculado no treino.
        Só pode ser chamado em um DataReader de treino.
        """
        if self.split_category != "train":
            raise RuntimeError(
                "mase_scale só pode ser chamado em DataReader('train')."
            )
        _validate_series_name(series_name)
        y = self.get_series(series_name)
        naive = y.shift(season)
        return float((y - naive).abs().dropna().mean())

    def prediction_template(self) -> pd.DataFrame:
        """
        DataFrame vazio com (date, series) para a validação.
        Só pode ser chamado em DataReader('validation').
        """
        if self.split_category != "validation":
            raise RuntimeError(
                "prediction_template só pode ser chamado em "
                "DataReader('validation')."
            )
        datas = self.get_dates()
        return pd.DataFrame(
            [(d, s) for d in datas for s in SERIES],
            columns=["date", "series"],
        )


def validate_all_splits() -> None:
    """
    Verifica invariantes ENTRE splits. Rode no início do run.py.

    Garante:
        - treino, validação e holdout não se sobrepõem;
        - treino termina antes da validação começar;
        - validação termina antes do holdout começar;
        - ordem cronológica global.
    """
    treino = DataReader("train")
    valid = DataReader("validation")
    holdout = DataReader("holdout")

    t = treino.get_dates()
    v = valid.get_dates()
    h = holdout.get_dates()

    if not t.max() < v.min():
        raise ValueError(
            f"treino invade validação: treino.max={t.max().date()}, "
            f"valid.min={v.min().date()}."
        )
    if not v.max() < h.min():
        raise ValueError(
            f"validação invade holdout: valid.max={v.max().date()}, "
            f"holdout.min={h.min().date()}."
        )
    if set(t) & set(v) or set(v) & set(h) or set(t) & set(h):
        raise ValueError("splits têm datas em comum.")