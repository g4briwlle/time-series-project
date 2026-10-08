import numpy as np
import pandas as pd

from src.data._config import BASE_PATH, VALIDATION_START, VALIDATION_END
from src.data.reader import DataReader, SERIES


def mae(y_real, y_pred):
    return float(np.mean(np.abs(np.asarray(y_real) - np.asarray(y_pred))))


def rmse(y_real, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_real) - np.asarray(y_pred)) ** 2)))


def mase(y_real, y_pred, escala):
    if not np.isfinite(escala) or escala <= 0:
        raise ValueError("A escala do MASE deve ser finita e maior que zero.")
    return mae(y_real, y_pred) / escala


def avaliar_previsoes(previsoes, validacao, escalas):
    """Recebe previsões (date, series, modelo, yhat), valores reais
    (date, series, y) e escalas de treino (series, mae).
    Também aceita DataReader('validation') para os valores reais.
    """
    if isinstance(validacao, DataReader):
        if validacao.split_category != "validation":
            raise ValueError("Use os valores reais da validação.")
        validacao = validacao.to_long()

    p = previsoes[["date", "series", "modelo", "yhat"]].copy()
    v = validacao[["date", "series", "y"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    v["date"] = pd.to_datetime(v["date"])
    escala = escalas.set_index("series", verify_integrity=True)["mae"]

    # Todas as séries devem ter os 28 dias, sem duplicatas.
    datas = pd.date_range(VALIDATION_START, VALIDATION_END)
    esperado = {(d, s) for d in datas for s in SERIES}

    # Alinha por data e série, independentemente da ordem dos arquivos.
    dados = p.merge(v, on=["date", "series"], validate="many_to_one")
    if not np.isfinite(dados[["y", "yhat"]].to_numpy(dtype=float)).all():
        raise ValueError("Valores reais ou previsões contêm NaN ou infinito.")

    linhas = []
    for (serie, modelo), grupo in dados.groupby(["series", "modelo"]):
        real = grupo["y"].to_numpy(dtype=float)
        pred = grupo["yhat"].to_numpy(dtype=float)
        linhas.append({
            "series": serie,
            "modelo": modelo,
            "mae": mae(real, pred),
            "rmse": rmse(real, pred),
            "mase": mase(real, pred, float(escala.loc[serie])),
        })
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    metricas = avaliar_previsoes(
        pd.read_csv(BASE_PATH / "outputs" / "previsoes_baselines.csv"),
        DataReader("validation"),
        pd.read_csv(BASE_PATH / "outputs" / "naive_sazonal_insample.csv"),
    )
    metricas.to_csv(BASE_PATH / "metricas.csv", index=False)
    print(metricas.to_string(index=False))
