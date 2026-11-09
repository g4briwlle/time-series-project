"""
Gráficos da entrega de árvore e boosting, a partir dos arquivos da raiz:

    python part3/src/plot_arvore_diagnostico.py

Gera em part3/figures:
  metricas_por_modelo.png        barras de MAE, RMSE e MASE por série e modelo
  previsoes_arvore_boosting.png  valor real e previsões na validação, por série
  residuos_arvore_boosting.png   resíduos da validação (y - yhat) no tempo e ACF, por série
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

RAIZ = Path(__file__).resolve().parents[2]
FIGURAS = RAIZ / "part3" / "figures"
SERIES = ["FOODS", "HOBBIES", "store_total"]
ARVORES = ["arvore", "boosting"]
ORDEM = ["media", "naive", "drift", "naive_sazonal", "sarima", "linear", "arvore", "boosting"]
COR = {
    "media": "#B8B8B8", "naive": "#9A9A9A", "drift": "#7C7C7C", "naive_sazonal": "#5E5E5E",
    "sarima": "#4C78A8", "linear": "#72B7B2", "arvore": "#F58518", "boosting": "#E45756",
}


def ler():
    previsoes = pd.read_csv(RAIZ / "previsoes_validacao.csv", parse_dates=["date"])
    metricas = pd.read_csv(RAIZ / "metricas.csv")
    real = pd.read_csv(RAIZ / "dados" / "validacao.csv", parse_dates=["date"]).set_index("date")
    return previsoes, metricas, real


def barras(metricas):
    fig, eixos = plt.subplots(3, 3, figsize=(15, 10))
    for j, serie in enumerate(SERIES):
        m = metricas[metricas["series"] == serie].set_index("modelo").reindex(ORDEM).dropna(how="all")
        for i, metrica in enumerate(["mae", "rmse", "mase"]):
            ax = eixos[i, j]
            ax.bar(m.index, m[metrica], color=[COR[k] for k in m.index])
            if metrica == "mase":
                ax.axhline(1.0, color="black", lw=0.8, ls="--")
            ax.set_title(f"{serie}: {metrica.upper()}")
            ax.tick_params(axis="x", rotation=45)
            for rot in ax.get_xticklabels():
                rot.set_ha("right")
    fig.suptitle("Métricas na validação (2016-03-28 a 2016-04-24); tracejado = MASE 1 (naive sazonal)")
    fig.tight_layout()
    fig.savefig(FIGURAS / "metricas_por_modelo.png", dpi=120)
    plt.close(fig)


def previsoes(prev, real):
    fig, eixos = plt.subplots(3, 1, figsize=(12, 11))
    for ax, serie in zip(eixos, SERIES):
        y = real[serie].loc["2016-03-28":"2016-04-24"]
        ax.plot(y.index, y.values, color="black", lw=2, label="real")
        for modelo in ["sarima", "linear", *ARVORES]:
            g = prev[(prev["series"] == serie) & (prev["modelo"] == modelo)].sort_values("date")
            if not g.empty:
                ax.plot(g["date"], g["yhat"], color=COR[modelo], label=modelo, alpha=0.9)
        ax.set_title(serie)
        ax.legend(loc="upper left", ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURAS / "previsoes_arvore_boosting.png", dpi=120)
    plt.close(fig)


def residuos(prev, real):
    fig, eixos = plt.subplots(3, 3, figsize=(15, 10))
    for i, serie in enumerate(SERIES):
        y = real[serie].loc["2016-03-28":"2016-04-24"]
        res = {}
        for modelo in ARVORES:
            g = prev[(prev["series"] == serie) & (prev["modelo"] == modelo)].set_index("date")["yhat"]
            res[modelo] = (y - g.reindex(y.index)).dropna()
        ax = eixos[i, 0]
        for modelo, r in res.items():
            ax.plot(r.index, r.values, color=COR[modelo], label=f"{modelo} (média {r.mean():.0f})")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{serie}: resíduo (real - previsto)")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
        for j, modelo in enumerate(ARVORES, start=1):
            r = res[modelo]
            plot_acf(r.values, lags=min(10, len(r) // 2 - 1), ax=eixos[i, j], color=COR[modelo])
            eixos[i, j].set_title(f"{serie}: ACF dos resíduos ({modelo})")
    fig.suptitle("Resíduos de árvore e boosting na validação (28 dias)")
    fig.tight_layout()
    fig.savefig(FIGURAS / "residuos_arvore_boosting.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    FIGURAS.mkdir(parents=True, exist_ok=True)
    prev, metricas, real = ler()
    barras(metricas)
    previsoes(prev, real)
    residuos(prev, real)
    print("figuras em", FIGURAS)
