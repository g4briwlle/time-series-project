"""
Script final do projeto: gera previsoes_validacao.csv e metricas.csv na raiz.

    python run.py

As previsões da Task 1 (quatro baselines e SARIMA) entram como foram entregues,
em part1/. As previsões de cada modelo das Tasks 2 e 3 são os CSVs versionados
que o autor de cada série gerou com o próprio script (ver a tabela do README).
As métricas são sempre recalculadas aqui a partir do yhat, com a escala do MASE
vinda do treino (MAE in-sample do naive sazonal, período 7).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_PATH = Path(__file__).resolve().parent
PART1 = BASE_PATH / "part1"
PART2 = BASE_PATH / "part2"
PART3 = BASE_PATH / "part3"

sys.path.insert(0, str(PART1))  # `src` é o pacote da Task 1

from src.baselines import naive_sazonal_insample
from src.data._config import VALIDATION_END, VALIDATION_START
from src.data.reader import SERIES, DataReader, validate_all_splits
from src.metrics import avaliar_previsoes

PREDICTIONS_PATH = BASE_PATH / "previsoes_validacao.csv"
METRICS_PATH = BASE_PATH / "metricas.csv"

PREDICTIONS_COLUMNS = ["date", "series", "modelo", "yhat"]

# Task 1 como foi entregue: baselines e SARIMA. Não se refaz. Os CSVs de
# part1/outputs são os que geraram part1/metricas.csv (part1/previsoes_validacao.csv
# traz um SARIMA ajustado em outra instalação, com diferença de até 0,04 no yhat).
TASK1_PREDICTIONS_PATHS = [
    PART1 / "outputs" / "previsoes_baselines.csv",
    PART1 / "outputs" / "previsoes_sarima.csv",
]
TASK1_METRICS_PATH = PART1 / "metricas.csv"

ARVORE = {"arvore", "boosting"}

# (classe, série, CSV de previsões, nomes de modelo aceitos nesse CSV).
# Em HOBBIES entra a estratégia direta; os arquivos `recursivo_` foram o teste da
# outra estratégia e não entram. Um CSV que ainda não existe vira aviso no final.
MODEL_PREDICTIONS = [
    ("linear", "FOODS", PART2 / "projeto-task2" / "linear_foods_model" / "previsoes_validacao_foods.csv", {"linear"}),
    ("linear", "HOBBIES", PART2 / "outputs" / "previsoes_linear_direto_HOBBIES.csv", {"linear"}),
    ("linear", "store_total", PART2 / "outputs" / "previsoes_linear_store_total.csv", {"linear"}),
    ("arvore", "FOODS", PART3 / "outputs" / "previsoes_direto_FOODS.csv", ARVORE),
    ("arvore", "HOBBIES", PART3 / "outputs" / "previsoes_direto_HOBBIES.csv", ARVORE),
    ("arvore", "store_total", PART2 / "outputs" / "previsoes_arvore_direto_store_total.csv", {"arvore"}),
    ("arvore", "store_total", PART2 / "outputs" / "previsoes_boosting_direto_store_total.csv", {"boosting"}),
]


def assert_same_data():
    """Os modelos leem os dados do split da Task 1; os arquivos devem ser os mesmos."""
    for nome in ("treino.csv", "validacao.csv", "holdout_datas.csv"):
        for pasta in (PART2 / "dados", BASE_PATH / "dados"):
            if (PART1 / "dados" / nome).read_bytes() != (pasta / nome).read_bytes():
                raise ValueError(f"{pasta.relative_to(BASE_PATH)}/{nome} difere de part1/dados/{nome}.")


def assert_predictions_df(df: pd.DataFrame, origem: str, serie: str | None = None):
    """Uma previsão por data da validação, sem duplicatas, sem NaN."""
    for coluna in PREDICTIONS_COLUMNS:
        assert coluna in df.columns, f"{origem}: falta a coluna '{coluna}'."
    if serie is not None:
        assert (df["series"] == serie).all(), f"{origem}: esperava só a série '{serie}'."

    esperadas = pd.date_range(VALIDATION_START, VALIDATION_END)
    for (s, modelo), grupo in df.groupby(["series", "modelo"]):
        datas = pd.DatetimeIndex(grupo["date"])
        assert not datas.duplicated().any(), f"{origem}: datas duplicadas em {s}/{modelo}."
        assert datas.sort_values().equals(esperadas), (
            f"{origem}: {s}/{modelo} não cobre exatamente os 28 dias da validação."
        )
        assert np.isfinite(grupo["yhat"].to_numpy(dtype=float)).all(), (
            f"{origem}: yhat com NaN ou infinito em {s}/{modelo}."
        )


def read_task1_predictions() -> pd.DataFrame:
    previsoes = pd.concat(
        [pd.read_csv(caminho, parse_dates=["date"])[PREDICTIONS_COLUMNS] for caminho in TASK1_PREDICTIONS_PATHS],
        ignore_index=True,
    )
    assert_predictions_df(previsoes, "Task 1")
    return previsoes


def read_model_predictions() -> tuple[list[pd.DataFrame], list[str]]:
    """Lê os CSVs das Tasks 2 e 3. O `modelo` do CSV tem que ser um dos nomes aceitos."""
    previsoes, faltas = [], []
    for classe, serie, caminho, aceitos in MODEL_PREDICTIONS:
        if not caminho.is_file():
            faltas.append(f"{classe} em {serie}: {caminho.relative_to(BASE_PATH)}")
            continue
        df = pd.read_csv(caminho, parse_dates=["date"])[PREDICTIONS_COLUMNS]
        df["modelo"] = df["modelo"].str.strip().str.lower()
        nomes = set(df["modelo"])
        assert nomes and nomes <= aceitos, (
            f"{caminho.name}: esperava os modelos {sorted(aceitos)}, veio {sorted(nomes)}."
        )
        assert_predictions_df(df, caminho.name, serie)
        previsoes.append(df)
    return previsoes, faltas


def assert_task1_metrics_unchanged(metricas: pd.DataFrame):
    entregues = pd.read_csv(TASK1_METRICS_PATH)
    novas = metricas.merge(entregues, on=["series", "modelo"], suffixes=("", "_task1"))
    assert len(novas) == len(entregues), "faltam linhas da Task 1 em metricas.csv."
    for coluna in ("mae", "rmse", "mase"):
        assert np.allclose(novas[coluna], novas[f"{coluna}_task1"], rtol=1e-9), (
            f"{coluna} da Task 1 mudou em metricas.csv."
        )


def coverage_warnings(previsoes: pd.DataFrame) -> list[str]:
    """O check cobra linear e boosting/arvore em cada uma das três séries."""
    avisos = []
    for serie in SERIES:
        modelos = set(previsoes.loc[previsoes["series"] == serie, "modelo"])
        if "linear" not in modelos:
            avisos.append(f"{serie}: sem linear")
        if not modelos & ARVORE:
            avisos.append(f"{serie}: sem boosting ou arvore")
    return avisos


def generate_files() -> list[str]:
    modelos, faltas = read_model_predictions()
    previsoes = pd.concat([read_task1_predictions(), *modelos], ignore_index=True)
    assert not previsoes.duplicated(["date", "series", "modelo"]).any(), "previsões duplicadas."
    previsoes.to_csv(PREDICTIONS_PATH, index=False)

    escala = naive_sazonal_insample(DataReader("train"))
    metricas = avaliar_previsoes(previsoes, DataReader("validation"), escala)
    assert_task1_metrics_unchanged(metricas)
    metricas.to_csv(METRICS_PATH, index=False)

    print(metricas.to_string(index=False))
    return [*faltas, *coverage_warnings(previsoes)]


def main():
    validate_all_splits()
    assert_same_data()

    avisos = generate_files()
    if avisos:
        print("\nAVISO: faltam previsões (arquivos esperados e séries sem cobertura):")
        for aviso in avisos:
            print(f"  {aviso}")


if __name__ == "__main__":
    main()
