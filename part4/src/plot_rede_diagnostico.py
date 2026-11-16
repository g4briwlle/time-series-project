"""
Gráficos da entrega da rede, a partir dos arquivos da raiz:

    python part4/src/plot_rede_diagnostico.py

Gera em part4/figures:
  metricas_por_modelo.png  barras de MAE, RMSE e MASE por série e modelo, com a rede
  previsoes_rede.png       valor real e previsões dos baselines, do linear, da árvore, do boosting e da rede
  residuos_rede.png        resíduos da rede (com o SARIMA da Task 1 de referência) no tempo, ACF dos
                           resíduos da rede e resíduo médio por dia da semana, por série

Lê metricas.csv, previsoes_validacao.csv e dados/validacao.csv e não altera nenhum deles. Uma série sem
`rede` em previsoes_validacao.csv (HOBBIES, até a LSTM entrar) aparece com um aviso no painel. Depois de
rodar `python run.py` com a rede nova, basta rodar este script de novo.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

RAIZ = Path(__file__).resolve().parents[2]
FIGURAS = RAIZ / "part4" / "figures"
SERIES = ["FOODS", "HOBBIES", "store_total"]
INICIO, FIM = "2016-03-28", "2016-04-24"
ORDEM = ["media", "naive", "drift", "naive_sazonal", "sarima", "linear", "arvore", "boosting", "rede"]
COR = {
    "media": "#B8B8B8", "naive": "#9A9A9A", "drift": "#7C7C7C", "naive_sazonal": "#5E5E5E",
    "sarima": "#4C78A8", "linear": "#72B7B2", "arvore": "#F58518", "boosting": "#E45756",
    "rede": "#B279A2",
}
DIAS = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
AVISO = "rede ainda sem previsão nesta série\n(não está em previsoes_validacao.csv)"


def ler():
    previsoes = pd.read_csv(RAIZ / "previsoes_validacao.csv", parse_dates=["date"])
    metricas = pd.read_csv(RAIZ / "metricas.csv")
    real = pd.read_csv(RAIZ / "dados" / "validacao.csv", parse_dates=["date"]).set_index("date")
    return previsoes, metricas, real


def tem_rede(prev, serie):
    return not prev[(prev["series"] == serie) & (prev["modelo"] == "rede")].empty


def aviso(ax):
    ax.text(0.5, 0.5, AVISO, transform=ax.transAxes, ha="center", va="center", fontsize=9, color=COR["rede"])
    ax.set_xticks([])
    ax.set_yticks([])


def residuo(prev, y, modelo):
    """Resíduo (real - previsto) de um modelo na validação, alinhado pelas datas do real."""
    g = prev[(prev["series"] == y.name) & (prev["modelo"] == modelo)].set_index("date")["yhat"]
    return (y - g.reindex(y.index)).dropna()


def barras(metricas):
    fig, eixos = plt.subplots(3, 3, figsize=(15, 10))
    for j, serie in enumerate(SERIES):
        m = metricas[metricas["series"] == serie].set_index("modelo").reindex(ORDEM).dropna(how="all")
        for i, metrica in enumerate(["mae", "rmse", "mase"]):
            ax = eixos[i, j]
            b = ax.bar(
                m.index, m[metrica], color=[COR[k] for k in m.index],
                edgecolor=["black" if k == "rede" else "none" for k in m.index], linewidth=1.3,
            )
            ax.bar_label(b, fmt="%.2f" if metrica == "mase" else "%.0f", fontsize=7)
            if metrica == "mase":
                ax.axhline(1.0, color="black", lw=0.8, ls="--")
            ax.set_title(f"{serie}: {metrica.upper()}")
            ax.margins(y=0.15)
            ax.tick_params(axis="x", rotation=45)
            for rot in ax.get_xticklabels():
                rot.set_ha("right")
            if i == 0 and "rede" not in m.index:
                ax.text(0.98, 0.95, "rede pendente", transform=ax.transAxes, ha="right", va="top",
                        fontsize=9, color=COR["rede"])
    fig.suptitle(
        "Métricas na validação (2016-03-28 a 2016-04-24); tracejado = MASE 1 (naive sazonal); "
        "rede com contorno preto"
    )
    fig.tight_layout()
    fig.savefig(FIGURAS / "metricas_por_modelo.png", dpi=120)
    plt.close(fig)


def previsoes(prev, real):
    fig, eixos = plt.subplots(3, 1, figsize=(12, 12))
    for ax, serie in zip(eixos, SERIES):
        y = real[serie].loc[INICIO:FIM]
        ax.plot(y.index, y.values, color="black", lw=2.2, label="real")
        for modelo in ["naive_sazonal", "sarima", "linear", "arvore", "boosting", "rede"]:
            g = prev[(prev["series"] == serie) & (prev["modelo"] == modelo)].sort_values("date")
            if g.empty:
                continue
            rede = modelo == "rede"
            ax.plot(
                g["date"], g["yhat"], color=COR[modelo], label=modelo,
                lw=2.4 if rede else 1.2, alpha=1.0 if rede else 0.75,
                ls="--" if modelo == "naive_sazonal" else "-",
            )
        ax.set_title(serie if tem_rede(prev, serie) else f"{serie} (rede ainda sem previsão)")
        ax.margins(y=0.25)
        ax.legend(loc="upper left", ncol=4, fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Valor real e previsões na validação (28 dias)")
    fig.tight_layout()
    fig.savefig(FIGURAS / "previsoes_rede.png", dpi=120)
    plt.close(fig)


def residuos(prev, real):
    fig, eixos = plt.subplots(3, 3, figsize=(15, 10))
    resumo = []
    for i, serie in enumerate(SERIES):
        ax_t, ax_acf, ax_dia = eixos[i]
        ax_t.set_title(f"{serie}: resíduo (real - previsto)")
        ax_acf.set_title(f"{serie}: ACF dos resíduos (rede)")
        ax_dia.set_title(f"{serie}: resíduo médio por dia da semana")
        if not tem_rede(prev, serie):
            for ax in eixos[i]:
                aviso(ax)
            resumo.append(f"{serie}: sem rede em previsoes_validacao.csv, painel com aviso")
            continue
        y = real[serie].loc[INICIO:FIM]
        res = {modelo: residuo(prev, y, modelo) for modelo in ["sarima", "rede"]}
        for modelo, lw in [("sarima", 1.2), ("rede", 2.4)]:
            r = res[modelo]
            ax_t.plot(r.index, r.values, color=COR[modelo], lw=lw, label=f"{modelo} (média {r.mean():.0f})")
        ax_t.axhline(0, color="black", lw=0.8)
        ax_t.legend(fontsize=8)
        ax_t.tick_params(axis="x", rotation=30)

        rede = res["rede"]
        plot_acf(rede.values, lags=min(10, len(rede) // 2 - 1), ax=ax_acf, color=COR["rede"])
        p = float(acorr_ljungbox(rede.values, lags=[7], return_df=True)["lb_pvalue"].iloc[0])
        ax_acf.set_title(f"{serie}: ACF dos resíduos (rede)\nLjung-Box lag 7: p = {p:.2f}")

        largura = 0.4
        for k, modelo in enumerate(["sarima", "rede"]):
            r = res[modelo]
            medio = r.groupby(r.index.dayofweek).mean().reindex(range(7))
            ax_dia.bar([d + (k - 0.5) * largura for d in range(7)], medio.values, width=largura,
                       color=COR[modelo], label=modelo)
        ax_dia.axhline(0, color="black", lw=0.8)
        ax_dia.set_xticks(range(7))
        ax_dia.set_xticklabels(DIAS)
        ax_dia.legend(fontsize=8)
        resumo.append(
            f"{serie}: viés da rede {rede.mean():.1f} (SARIMA {res['sarima'].mean():.1f}), "
            f"Ljung-Box lag 7 p = {p:.2f}"
        )
    fig.suptitle("Resíduos da rede na validação (28 dias), com o SARIMA da Task 1 de referência")
    fig.tight_layout()
    fig.savefig(FIGURAS / "residuos_rede.png", dpi=120)
    plt.close(fig)
    return resumo


if __name__ == "__main__":
    FIGURAS.mkdir(parents=True, exist_ok=True)
    prev, metricas, real = ler()
    barras(metricas)
    previsoes(prev, real)
    for linha in residuos(prev, real):
        print(linha)
    print("figuras em", FIGURAS)
