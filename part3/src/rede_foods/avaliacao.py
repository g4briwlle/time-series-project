from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare_data import RAIZ, carregar_dados
from src.metrics import avaliar_previsoes

PASTA = Path(__file__).resolve().parent
ROTULOS = {
    "naive_sazonal": "Naive semanal",
    "sarima": "SARIMA",
    "linear": "Lasso",
    "rede": "LSTM",
}
CORES = ["#8C8C8C", "#AA6100", "#4066A0", "#007A68"]

def carregar_previsoes(datas_validacao):
    referencias = pd.read_csv(RAIZ / "previsoes_validacao.csv", parse_dates=["date"])
    referencias = referencias.loc[
        referencias["series"].eq("FOODS")
        & referencias["modelo"].isin(["naive_sazonal", "sarima", "linear"])
    ]
    lstm = pd.read_csv(PASTA / "previsoes_validacao_foods.csv", parse_dates=["date"])
    if not (lstm["series"].eq("FOODS") & lstm["modelo"].eq("rede")).all():
        raise ValueError("O arquivo da LSTM deve conter somente previsões de FOODS.")

    previsoes = pd.concat([referencias, lstm], ignore_index=True)
    por_data = previsoes.pivot(index="date", columns="modelo", values="yhat")
    if not por_data.index.equals(datas_validacao) or set(por_data.columns) != set(ROTULOS):
        raise ValueError("São esperados os quatro modelos nas 28 datas da validação.")
    if not np.isfinite(por_data.to_numpy()).all():
        raise ValueError("Cada modelo deve ter uma previsão válida por dia.")
    return previsoes, por_data[list(ROTULOS)]

def plotar_metricas(metricas):
    valores = metricas.set_index("modelo").loc[list(ROTULOS)].rename(index=ROTULOS)
    fig, eixos = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for eixo, metrica in zip(eixos, ["mae", "rmse", "mase"]):
        barras = eixo.bar(valores.index, valores[metrica], color=CORES)
        eixo.bar_label(barras, fmt="%.3f" if metrica == "mase" else "%.1f", padding=3)
        eixo.set_title(metrica.upper())
        eixo.margins(y=0.2)
        eixo.tick_params(axis="x", labelrotation=25)
    fig.suptitle("FOODS | Erro na validação oficial")
    fig.savefig(PASTA / "metricas_validacao.png", dpi=150)
    plt.close(fig)

def plotar_previsoes(reais, por_data, erros):
    fig, (vendas_ax, erros_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(11, 7), layout="constrained"
    )
    vendas_ax.plot(reais.index, reais, label="Observado", color="black", linewidth=2)
    for (modelo, rotulo), cor in zip(ROTULOS.items(), CORES):
        vendas_ax.plot(por_data.index, por_data[modelo], label=rotulo, color=cor)
    vendas_ax.set_title("FOODS | Vendas observadas e previsões")
    vendas_ax.set_ylabel("Unidades vendidas")
    vendas_ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.05))

    erros_ax.axhline(0, color="black", linewidth=0.8)
    erros_ax.plot(erros.index, erros["rede"], color=CORES[-1], marker=".", label="LSTM")
    domingos = erros.loc[erros.index.dayofweek == 6, "rede"]
    erros_ax.scatter(domingos.index, domingos, color="#AA6100", zorder=3, label="Domingos")
    erros_ax.set_ylabel("Observado − previsto")
    erros_ax.set_xlabel("Data")
    erros_ax.legend()
    fig.savefig(PASTA / "previsoes_e_erros.png", dpi=150)
    plt.close(fig)


treino, _, datas_validacao = carregar_dados()
reais = pd.read_csv(
    RAIZ / "dados" / "validacao.csv", parse_dates=["date"], usecols=["date", "FOODS"]
).set_index("date", verify_integrity=True)["FOODS"].sort_index()
if not reais.index.equals(datas_validacao) or not np.isfinite(reais).all():
    raise ValueError("A validação deve conter uma venda válida por dia no período oficial.")

previsoes, por_data = carregar_previsoes(datas_validacao)
validacao = reais.rename("y").reset_index().assign(series="FOODS")
escala = treino.diff(7).abs().mean()
escalas = pd.DataFrame({"series": ["FOODS"], "mae": [escala]})
metricas = avaliar_previsoes(previsoes, validacao, escalas)
metricas.to_csv(PASTA / "comparacao_validacao.csv", index=False)

erros = por_data.rsub(reais, axis="index")
erros.assign(y=reais).to_csv(PASTA / "erros_validacao.csv")
plotar_metricas(metricas)
plotar_previsoes(reais, por_data, erros)

erro = erros["rede"]
domingos = erro.loc[erro.index.dayofweek == 6]
pior_dia = erro.abs().idxmax()
print(metricas.to_string(index=False))
print(f"Escala do MASE: {escala:.2f}")
print(f"Erro médio da LSTM: {erro.mean():+.2f}")
print(f"Subestimações: {erro.gt(0).sum()} | Superestimações: {erro.lt(0).sum()}")
print(f"Domingos: {domingos.gt(0).sum()}/{len(domingos)} subestimados | erro médio {domingos.mean():+.2f}")
print(f"Maior erro absoluto: {pior_dia:%d/%m/%Y} | erro {erro[pior_dia]:+.2f}")
