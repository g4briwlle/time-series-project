import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "part1"))

from src.data._config import TRAIN_END, TRAIN_START, VALIDATION_END, VALIDATION_START

HORIZONTE = 28
CORTES_INTERNOS = 3
EVENTOS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
NUMERICAS = ["snap_CA", "preco_FOODS"]
CATEGORICAS = ["dia_semana", "mes", *EVENTOS]


def carregar_dados():
    pasta = RAIZ / "dados"
    vendas = pd.read_csv(
        pasta / "treino.csv", usecols=["date", "FOODS"], parse_dates=["date"]
    ).set_index("date", verify_integrity=True)["FOODS"].sort_index()
    covariaveis = pd.read_csv(
        pasta / "covariaveis.csv",
        usecols=["date", *NUMERICAS, *EVENTOS],
        parse_dates=["date"],
    ).set_index("date", verify_integrity=True)

    datas_treino = pd.date_range(TRAIN_START, TRAIN_END)
    datas_validacao = pd.date_range(VALIDATION_START, VALIDATION_END)
    if not vendas.index.equals(datas_treino) or not np.isfinite(vendas).all():
        raise ValueError("O treino de FOODS deve conter uma venda válida por dia até a origem.")

    datas = datas_treino.append(datas_validacao)
    if not datas.isin(covariaveis.index).all():
        raise ValueError("Há datas sem covariáveis de FOODS.")
    entradas = covariaveis.loc[datas].copy()
    if not np.isfinite(entradas[NUMERICAS].to_numpy()).all():
        raise ValueError("SNAP ou preço ausente no período de treino ou previsão.")

    entradas["dia_semana"] = datas.dayofweek
    entradas["mes"] = datas.month
    entradas[EVENTOS] = entradas[EVENTOS].fillna("sem_evento")
    return vendas, entradas, datas_validacao

def preparar_corte(vendas, entradas, datas_previsao):

    transformador = ColumnTransformer([
        ("categorias", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
        ("preco", StandardScaler(), ["preco_FOODS"]),
        ("snap", "passthrough", ["snap_CA"]),
    ])
    cov_treino = transformador.fit_transform(entradas.loc[vendas.index])
    cov_previsao = transformador.transform(entradas.loc[datas_previsao])

    media, desvio = vendas.mean(), vendas.std(ddof=0)
    if desvio <= 0 or not np.isfinite(desvio):
        raise ValueError("O desvio padrão das vendas de treino deve ser positivo e finito.")
    y = ((vendas.to_numpy() - media) / desvio).astype(np.float32)
    return y, cov_treino, cov_previsao, media, desvio


vendas, entradas, datas_validacao = carregar_dados()
cortes = TimeSeriesSplit(n_splits=CORTES_INTERNOS, test_size=HORIZONTE)

for corte, (i_treino, i_previsao) in enumerate(cortes.split(vendas), 1):
    treino = vendas.iloc[i_treino]
    datas = vendas.index[i_previsao]
    _, x_treino, x_previsao, _, _ = preparar_corte(treino, entradas, datas)
    print(
        f"Corte {corte}: treino até {treino.index[-1]:%d/%m/%Y} "
        f"| covariáveis de treino {x_treino.shape} e previsão {x_previsao.shape}"
    )

_, x_treino, x_previsao, media, desvio = preparar_corte(vendas, entradas, datas_validacao)
print(
    f"FOODS: {len(vendas)} vendas de treino, {len(datas_validacao)} datas de previsão, "
    f"{x_treino.shape[1]} covariáveis após codificação."
)
print(f"Vendas no treino: média={media:.2f}, desvio padrão={desvio:.2f}.")
