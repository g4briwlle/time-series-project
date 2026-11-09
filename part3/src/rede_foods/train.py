from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit
from torch import nn

from prepare_data import CORTES_INTERNOS, HORIZONTE, carregar_dados, preparar_corte
from main import HISTORICOS, RedeFoods, janelas_treino

PASTA = Path(__file__).resolve().parent
EPOCAS = (20, 40)
SEMENTE = 42
TAMANHO_LOTE = 128

def treinar(y, cov_treino, historico, epocas_avaliadas):
    """Entrega a rede nos pontos de avaliação de um mesmo treinamento."""
    torch.manual_seed(SEMENTE)
    modelo = RedeFoods(cov_treino.shape[1])
    otimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
    perda_absoluta = nn.L1Loss()
    passado, covariaveis, alvos = janelas_treino(y, cov_treino, historico)

    for epoca in range(1, max(epocas_avaliadas) + 1):
        modelo.train()
        for inicio in range(0, len(alvos), TAMANHO_LOTE):
            lote = slice(inicio, inicio + TAMANHO_LOTE)
            otimizador.zero_grad()
            previstos = modelo(passado[lote], covariaveis[lote])
            perda = perda_absoluta(previstos, alvos[lote])
            perda.backward()
            otimizador.step()

        if epoca in epocas_avaliadas:
            yield epoca, modelo


@torch.inference_mode()
def prever(modelo, y, cov_previsao, historico, media, desvio):
    """Prevê os 28 dias em unidades de venda a partir de uma só origem."""
    modelo.eval()
    passado = torch.from_numpy(y[-historico:]).reshape(1, historico, 1)
    futuro = torch.as_tensor(cov_previsao, dtype=torch.float32).unsqueeze(0)
    padronizadas = modelo(passado, futuro).squeeze(0).numpy()
    return np.maximum(padronizadas * desvio + media, 0)

def selecionar_configuracao(vendas, entradas):
    resultados = []
    cortes = TimeSeriesSplit(n_splits=CORTES_INTERNOS, test_size=HORIZONTE)
    for corte, (i_treino, i_avaliacao) in enumerate(cortes.split(vendas), 1):
        treino = vendas.iloc[i_treino]
        avaliacao = vendas.iloc[i_avaliacao]
        y, cov_treino, cov_previsao, media, desvio = preparar_corte(
            treino, entradas, avaliacao.index
        )
        for historico in HISTORICOS:
            for epoca, modelo in treinar(y, cov_treino, historico, EPOCAS):
                yhat = prever(modelo, y, cov_previsao, historico, media, desvio)
                resultados.append({
                    "corte": corte,
                    "origem": treino.index[-1].date(),
                    "historico": historico,
                    "epocas": epoca,
                    "mae": np.abs(avaliacao.to_numpy() - yhat).mean(),
                })
        print(f"Corte {corte}: origem em {treino.index[-1]:%d/%m/%Y}.", flush=True)

    detalhes = pd.DataFrame(resultados)
    resumo = (
        detalhes.groupby(["historico", "epocas"], as_index=False)
        .agg(mae_medio=("mae", "mean"), mae_desvio=("mae", "std"))
        .sort_values(["mae_medio", "historico", "epocas"])
    )
    detalhes.to_csv(PASTA / "comparacao_cortes.csv", index=False)
    resumo.to_csv(PASTA / "comparacao_modelos.csv", index=False)
    print(resumo.to_string(index=False))
    melhor = resumo.iloc[0]
    return int(melhor["historico"]), int(melhor["epocas"])


torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

vendas, entradas, datas_validacao = carregar_dados()
historico, epocas = selecionar_configuracao(vendas, entradas)
print(f"Configuração escolhida: {historico} dias e {epocas} épocas.", flush=True)

y, cov_treino, cov_previsao, media, desvio = preparar_corte(
    vendas, entradas, datas_validacao
)
_, modelo = next(treinar(y, cov_treino, historico, (epocas,)))
yhat = prever(modelo, y, cov_previsao, historico, media, desvio)

previsoes = pd.DataFrame({
    "date": datas_validacao,
    "series": "FOODS",
    "modelo": "rede",
    "yhat": yhat.astype(float),
})
previsoes.to_csv(PASTA / "previsoes_validacao_foods.csv", index=False)
print(f"Previsões de FOODS geradas: {len(previsoes)} dias.")
