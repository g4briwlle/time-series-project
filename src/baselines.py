# src/baselines.py
import numpy as np
import pandas as pd

from src.data._config import BASE_PATH
from src.data.reader import DataReader, SERIES, validate_all_splits

SEASON = 7
OUTPUT_PATH = BASE_PATH / "outputs"


# ---------- Métodos ----------

def media(y: pd.Series, h: int) -> np.ndarray:
    """yhat_{T+h} = média de y_1, ..., y_T."""
    return np.full(h, y.mean())


def naive(y: pd.Series, h: int) -> np.ndarray:
    """yhat_{T+h} = y_T."""
    return np.full(h, float(y.iloc[-1]))


def naive_sazonal(y: pd.Series, h: int, m: int = SEASON) -> np.ndarray:
    """yhat_{T+h} = y_{T+h-m(k+1)}, com k = (h-1) // m: repete o último ciclo de m dias."""
    ultimo_ciclo = y.iloc[-m:].to_numpy(dtype=float)
    return np.tile(ultimo_ciclo, h // m + 1)[:h]


def drift(y: pd.Series, h: int) -> np.ndarray:
    """yhat_{T+h} = y_T + h (y_T - y_1) / (T - 1)."""
    inclinacao = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
    return y.iloc[-1] + inclinacao * np.arange(1, h + 1)


BASELINES = {
    "media": media,
    "naive": naive,
    "naive_sazonal": naive_sazonal,
    "drift": drift,
}


# ---------- Saídas ----------

def prever_baselines(train: DataReader, datas: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Previsões dos quatro baselines para as três séries nas `datas` da
    validação, no formato de previsoes_validacao.csv: colunas 'date',
    'series', 'modelo', 'yhat'. Da validação entram só as datas, nunca o y.
    """
    # naive sazonal e drift contam h a partir do último dia do treino
    if datas[0] != train.get_dates()[-1] + pd.Timedelta(days=1):
        raise ValueError("as datas devem começar no dia seguinte ao fim do treino.")

    previsoes = []
    for nome in SERIES:
        y = train.get_series(nome)
        for modelo, prever in BASELINES.items():
            previsoes.append(pd.DataFrame({
                "date": datas,
                "series": nome,
                "modelo": modelo,
                "yhat": prever(y, len(datas)),
            }))
    return pd.concat(previsoes, ignore_index=True)


def naive_sazonal_insample(train: DataReader, m: int = SEASON) -> pd.DataFrame:
    """
    MAE e RMSE in-sample do naive sazonal no treino (y_t contra y_{t-m}).
    O 'mae' é a escala do MASE: MASE = MAE na validação / mae.

    Os zeros de Natal ficam no cálculo, como no check_task1.py; tirá-los
    muda a escala e o MASE deixa de bater com o recálculo do professor.
    """
    linhas = []
    for nome in SERIES:
        y = train.get_series(nome)
        erro = (y - y.shift(m)).dropna()
        linhas.append({
            "series": nome,
            "mae": erro.abs().mean(),
            "rmse": np.sqrt((erro ** 2).mean()),
        })
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    validate_all_splits()
    train = DataReader("train")
    valid = DataReader("validation")

    OUTPUT_PATH.mkdir(exist_ok=True)
    prever_baselines(train, valid.get_dates()).to_csv(OUTPUT_PATH / "previsoes_baselines.csv", index=False)
    naive_sazonal_insample(train).to_csv(OUTPUT_PATH / "naive_sazonal_insample.csv", index=False)
    print(f"Previsões dos baselines e escala do MASE salvas em {OUTPUT_PATH}")
