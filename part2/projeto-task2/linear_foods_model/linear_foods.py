from pathlib import Path
import pandas as pd
import numpy as np

#import para preparacao
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


#import para modelo
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

pasta_script = Path(__file__).resolve().parent
datos = pasta_script.parent.parent / "dados"

treino = pd.read_csv(
    datos / "treino.csv",
    parse_dates=["date"],
)[["date", "FOODS"]]

covariaveis = pd.read_csv(
    datos / "covariaveis.csv",
    parse_dates=["date"],
)[[
    "date",
    "snap_CA",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "preco_FOODS",
]]

base_treino = treino.merge(
    covariaveis,
    on="date",
    how="left",
    validate="one_to_one",
)

columnas_eventos = [
    "event_name_1", "event_type_1",
    "event_name_2", "event_type_2",
]


def preparar_entradas(base):
    # Preparar as entradas para o modelo sem modificar a base original
    entradas = base.copy()

    entradas["dia_semana"] = entradas["date"].dt.dayofweek
    entradas["mes"] = entradas["date"].dt.month

    columnas_eventos = [
        "event_name_1", "event_type_1",
        "event_name_2", "event_type_2",
    ]
    entradas[columnas_eventos] = (
        entradas[columnas_eventos].fillna("sem_evento")
    )

    return entradas


base = preparar_entradas(base_treino)


categoricas = [
    "dia_semana",
    "mes",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]

columnas_entrada = categoricas + ["preco_FOODS", "snap_CA"]

base = base.sort_values("date").reset_index(drop=True)

X_treino = base[columnas_entrada]
y_treino = base["FOODS"]



# Verificacao de tempo e valores nulos
assert base["date"].max() <= pd.Timestamp("2016-03-27")
assert not X_treino.isna().any().any()
assert not y_treino.isna().any()

# definir las transformaciones
preprocesamiento = ColumnTransformer(
    transformers=[
        (
            "categoricas",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categoricas,
        ),
        ("precio", StandardScaler(), ["preco_FOODS"]),
        ("snap", "passthrough", ["snap_CA"]),
    ]
)


candidatos = [
    {
        "nombre": "linear",
        "alpha": None,
        "estimador": LinearRegression(),
    }
]

for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    candidatos.append({
        "nombre": "ridge",
        "alpha": alpha,
        "estimador": Ridge(alpha=alpha),
    })

    candidatos.append({
        "nombre": "lasso",
        "alpha": alpha,
        "estimador": Lasso(
            alpha=alpha,
            max_iter=100_000,
        ),
    })

# Em cada corte:
# - treinamento - todos os dias anteriores
# - avaliacao interna - os 28 dias seguintes
divisor = TimeSeriesSplit(n_splits=3, test_size=28)
cortes = list(divisor.split(X_treino))

print("\nCortes de validación interna:")

for numero, (indices_treino, indices_validacao) in enumerate(
    cortes, start=1
):
    fechas_treino = base.iloc[indices_treino]["date"]
    fechas_validacao = base.iloc[indices_validacao]["date"]

    print(
        f"Corte {numero}: "
        f"treino {fechas_treino.min().date()} "
        f"a {fechas_treino.max().date()} | "
        f"avaliação {fechas_validacao.min().date()} "
        f"a {fechas_validacao.max().date()}"
    )


resultados = []

for candidato_id, candidato in enumerate(candidatos):
    for numero, (indices_treino, indices_validacao) in enumerate(
        cortes, start=1
    ):
        X_ajuste = X_treino.iloc[indices_treino]
        y_ajuste = y_treino.iloc[indices_treino]

        X_interno = X_treino.iloc[indices_validacao]
        y_interno = y_treino.iloc[indices_validacao]

        # Pipeline novo
        pipeline = Pipeline([
            ("preprocesamiento", clone(preprocesamiento)),
            ("regresion", clone(candidato["estimador"])),
        ])

        pipeline.fit(X_ajuste, y_ajuste)
        previsoes = pipeline.predict(X_interno)

        resultados.append({
            "candidato_id": candidato_id,
            "modelo": candidato["nombre"],
            "alpha": candidato["alpha"],
            "corte": numero,
            "mae": mean_absolute_error(y_interno, previsoes),
        })

resultados = pd.DataFrame(resultados)

# MAE promedio dos cortes
resumo = (
    resultados
    .groupby(
        ["candidato_id", "modelo", "alpha"],
        dropna=False,
    )
    .agg(
        mae_medio=("mae", "mean"),
        mae_desvio=("mae", "std"),
    )
    .reset_index()
    .sort_values(
        ["mae_medio", "candidato_id"],
        kind="stable",
    )
    .reset_index(drop=True)
)

print("\nComparação interna — menor MAE é melhor:")
print(
    resumo[["modelo", "alpha", "mae_medio", "mae_desvio"]]
    .to_string(index=False)
)

# escolher o ganhador
melhor_id = int(resumo.iloc[0]["candidato_id"])
melhor = candidatos[melhor_id]

print(f"\nModelo escolhido: {melhor['nombre']}")

if melhor["alpha"] is not None:
    print(f"Alpha escolhido: {melhor['alpha']}")

print(f"MAE médio interno: {resumo.iloc[0]['mae_medio']:.2f}")

# Treinamento com o escolhido
modelo_foods = Pipeline([
    ("preprocesamiento", clone(preprocesamiento)),
    ("regresion", clone(melhor["estimador"])),
])

modelo_foods.fit(X_treino, y_treino)

print(f"\nModelo final treinado com {len(X_treino)} dias.")
print(
    "Período:",
    base["date"].min().date(),
    "a",
    base["date"].max().date(),
)

pasta_comparacao = pasta_script / "comparacao_de_modelos"
pasta_comparacao.mkdir(exist_ok=True)

resultados.to_csv(
    pasta_comparacao / "foods_comparacao_cortes.csv",
    index=False,
)
resumo.to_csv(
    pasta_comparacao / "foods_comparacao_modelos.csv",
    index=False,
)

#--------------- 28 predicoes ----------------

datas_validacao = pd.DataFrame({
    "date": pd.date_range("2016-03-28", "2016-04-24")
})

base_validacao = datas_validacao.merge(
    covariaveis,
    on="date",
    how="left",
    validate="one_to_one",
    indicator=True,
)


base_validacao = base_validacao.drop(columns="_merge")

# mesma preparacao do treino
base_validacao = preparar_entradas(base_validacao)
X_validacao = base_validacao[columnas_entrada]


yhat = modelo_foods.predict(X_validacao)

previsoes_foods = pd.DataFrame({
    "date": base_validacao["date"],
    "series": "FOODS",
    "modelo": "Linear",
    "yhat": yhat,
})


previsoes_foods.to_csv(
    pasta_script / "previsoes_validacao_foods.csv",
    index=False,
)

print("\nPrevisões de FOODS:")
print(previsoes_foods.to_string(index=False)) 

#----------------------- métricas de avaliação -------------------

validacao = pd.read_csv(
    datos / "validacao.csv",
    parse_dates=["date"],
)[["date", "FOODS"]]

avaliacao = previsoes_foods.merge(
    validacao,
    on="date",
    how="left",
    validate="one_to_one",
)

assert len(avaliacao) == 28
assert np.isfinite(
    avaliacao[["FOODS", "yhat"]].to_numpy(dtype=float)
).all()

erros = avaliacao["FOODS"] - avaliacao["yhat"]

mae = float(erros.abs().mean())
rmse = float(np.sqrt((erros ** 2).mean()))

# MASE semanal do treinamento
escala_mase = float(
    y_treino.diff(7).abs().dropna().mean()
)

assert np.isfinite(escala_mase) and escala_mase > 0

mase = mae / escala_mase

metricas_foods = pd.DataFrame([{
    "series": "FOODS",
    "modelo": "Linear",
    "mae": mae,
    "rmse": rmse,
    "mase": mase,
}])

metricas_foods.to_csv(
    pasta_script / "metricas_foods.csv",
    index=False,
)

print("\nMétricas na validação oficial:")
print(metricas_foods.to_string(index=False))
print(f"\nEscala do MASE: {escala_mase:.6f}")

