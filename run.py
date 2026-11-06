"""
Script final do projeto: gera previsoes_validacao.csv e metricas.csv na raiz.

    python run.py                    refaz os modelos da Task 2 e consolida
    python run.py --so-consolidar    consolida os CSVs de previsões que já existem

As previsões da Task 1 (quatro baselines e SARIMA) entram como foram entregues,
em part1/. As métricas são sempre recalculadas aqui a partir do yhat, com a
escala do MASE vinda do treino (MAE in-sample do naive sazonal, período 7).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_PATH = Path(__file__).resolve().parent
PART1 = BASE_PATH / "part1"
PART2 = BASE_PATH / "part2"

sys.path.insert(0, str(PART1))  # `src` é o pacote da Task 1

from src.baselines import naive_sazonal_insample
from src.data._config import VALIDATION_END, VALIDATION_START
from src.data.reader import DataReader, validate_all_splits
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

# Scripts que geram as previsões da Task 2 e, para o pacote com imports
# relativos (store_total), o módulo rodado da raiz.
MODEL_SCRIPTS = [
    PART2 / "src" / "model_direto_hobbies.py",
    PART2 / "projeto-task2" / "linear_foods_model" / "linear_foods.py",
]
MODEL_MODULES = [
    "part2.src.model_store_total.model_store_total",
]

# CSV de previsões de cada classe, por série. No HOBBIES entra a estratégia
# direta; os arquivos `recursivo_` não entram.
TASK2_PREDICTIONS_PATHS = {
    "linear": {
        "FOODS": PART2 / "projeto-task2" / "linear_foods_model" / "previsoes_validacao_foods.csv",
        "HOBBIES": PART2 / "outputs" / "previsoes_linear_direto_HOBBIES.csv",
        "store_total": PART2 / "outputs" / "previsoes_linear_store_total.csv",
    },
}


def assert_same_data():
    """Os modelos da Task 2 leem part2/dados; os arquivos devem ser os do split da Task 1."""
    for nome in ("treino.csv", "validacao.csv", "holdout_datas.csv"):
        if (PART1 / "dados" / nome).read_bytes() != (PART2 / "dados" / nome).read_bytes():
            raise ValueError(f"part2/dados/{nome} difere de part1/dados/{nome}.")


def run_model_scripts():
    comandos = [(script.relative_to(BASE_PATH), [str(script)]) for script in MODEL_SCRIPTS]
    comandos += [(modulo, ["-m", modulo]) for modulo in MODEL_MODULES]
    for nome, argumentos in comandos:
        print(f"Rodando {nome}")
        subprocess.run([sys.executable, *argumentos], cwd=BASE_PATH, check=True)


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


def read_task2_predictions() -> tuple[list[pd.DataFrame], list[str]]:
    """Lê uma classe por série. O nome do modelo no CSV final é o da classe."""
    previsoes, faltas = [], []
    for classe, por_serie in TASK2_PREDICTIONS_PATHS.items():
        for serie, caminho in por_serie.items():
            if not caminho.is_file():
                faltas.append(f"{classe} em {serie}: {caminho.relative_to(BASE_PATH)}")
                continue
            df = pd.read_csv(caminho, parse_dates=["date"])[PREDICTIONS_COLUMNS]
            assert df["modelo"].str.strip().str.lower().eq(classe).all(), (
                f"{caminho.name}: esperava o modelo '{classe}'."
            )
            df["modelo"] = classe
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


def generate_files() -> list[str]:
    task2, faltas = read_task2_predictions()
    previsoes = pd.concat([read_task1_predictions(), *task2], ignore_index=True)
    assert not previsoes.duplicated(["date", "series", "modelo"]).any(), "previsões duplicadas."
    previsoes.to_csv(PREDICTIONS_PATH, index=False)

    escala = naive_sazonal_insample(DataReader("train"))
    metricas = avaliar_previsoes(previsoes, DataReader("validation"), escala)
    assert_task1_metrics_unchanged(metricas)
    metricas.to_csv(METRICS_PATH, index=False)

    print(metricas.to_string(index=False))
    return faltas


def main():
    parser = argparse.ArgumentParser(description="Gera previsoes_validacao.csv e metricas.csv.")
    parser.add_argument(
        "--so-consolidar",
        action="store_true",
        help="não refaz os modelos da Task 2; usa os CSVs de previsões que já existem",
    )
    args = parser.parse_args()

    validate_all_splits()
    assert_same_data()
    if not args.so_consolidar:
        run_model_scripts()

    faltas = generate_files()
    if faltas:
        print("\nAVISO: faltam previsões da Task 2:")
        for falta in faltas:
            print(f"  {falta}")


if __name__ == "__main__":
    main()
