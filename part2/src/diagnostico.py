"""
Resíduos na validação (y - yhat) e métricas por modelo, para as classes já entregues.

Lê previsoes_validacao.csv e metricas.csv da raiz (gerados pelo run.py) e grava
as figuras em part2/figures. Da raiz do repo:

    python part2/src/diagnostico.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_PATH / "part1"))  # `src` é o pacote da Task 1

from src.data.reader import SERIES, DataReader

FIGURES_PATH = BASE_PATH / "part2" / "figures"
MODELOS = ["naive_sazonal", "sarima", "linear"]


def plot_residuos(previsoes: pd.DataFrame, validacao: pd.DataFrame):
    fig, axes = plt.subplots(len(SERIES), 1, figsize=(10, 9), sharex=True)
    for ax, serie in zip(axes, SERIES):
        y = validacao[validacao["series"] == serie].set_index("date")["y"]
        for modelo in MODELOS:
            yhat = previsoes[(previsoes["series"] == serie) & (previsoes["modelo"] == modelo)]
            if yhat.empty:
                continue
            ax.plot(y.index, y - yhat.set_index("date")["yhat"].reindex(y.index), label=modelo)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{serie}: y - yhat na validação")
        ax.legend(loc="upper left", ncol=len(MODELOS))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "residuos_validacao.png", dpi=120)
    plt.close(fig)


def plot_metricas(metricas: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metrica in zip(axes, ["mae", "rmse", "mase"]):
        tabela = metricas[metricas["modelo"].isin(MODELOS)].pivot(index="series", columns="modelo", values=metrica)
        tabela.reindex(SERIES)[[m for m in MODELOS if m in tabela.columns]].plot.bar(ax=ax, rot=0)
        ax.set_title(metrica.upper())
        ax.set_xlabel("")
    axes[2].axhline(1, color="black", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "metricas_por_modelo.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES_PATH.mkdir(exist_ok=True)
    previsoes = pd.read_csv(BASE_PATH / "previsoes_validacao.csv", parse_dates=["date"])
    metricas = pd.read_csv(BASE_PATH / "metricas.csv")

    plot_residuos(previsoes, DataReader("validation").to_long())
    plot_metricas(metricas)
    print(f"Figuras salvas em {FIGURES_PATH}")
