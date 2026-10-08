# Lendo dados com Data Reader

Para garantir o cumprimento do critério de split dos dados e evitar corrompimento deles, procure nunca acessar diretamente a pasta `dados`. Ao invés, use a classe `src.data.reader.DataReader`. Abaixo, alguns exemplos de como utilizar a classe para dados de treino e validação:

### Diagnóstico (ACF/PACF) — quem cuida do item 4

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from src.data.reader import DataReader, SERIES

train = DataReader("train")

for nome in SERIES:
    y = train.get_series(nome)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
    axes[0].plot(y.index, y.values)
    axes[0].set_title(f"{nome} — treino")
    plot_acf(y.dropna(), lags=40, ax=axes[1])
    plot_pacf(y.dropna(), lags=40, ax=axes[2], method="ywm")
    fig.tight_layout()
    fig.savefig(f"figuras/diagnostico_{nome}.png", dpi=120)
```

### Baselines (média, naive, naive sazonal e drift)

Implementados em `src/baselines.py`. Da raiz do repo:

```bash
python -m src.baselines
```

Isso gera em `outputs/`:

- `previsoes_baselines.csv`: previsões dos 28 dias de validação para as três séries, com as colunas de `previsoes_validacao.csv` (`date`, `series`, `modelo`, `yhat`) e `modelo` em `media`, `naive`, `naive_sazonal`, `drift`;
- `naive_sazonal_insample.csv`: MAE e RMSE in-sample do naive sazonal (m = 7) no treino. O `mae` é a escala do MASE.

Para usar no `run.py`:

```python
from src.baselines import prever_baselines, naive_sazonal_insample
from src.data.reader import DataReader

train = DataReader("train")
valid = DataReader("validation")

previsoes = prever_baselines(train, valid.get_dates())  # date, series, modelo, yhat
escala_mase = naive_sazonal_insample(train)             # series, mae, rmse
```

### ARIMA/SARIMA — quem cuida do modelo

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.data.reader import DataReader

train = DataReader("train")
valid = DataReader("validation")

y_train = train.get_series("store_total").dropna()
h = len(valid.get_dates())

modelo = SARIMAX(
    y_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
res = modelo.fit(disp=False)

yhat = res.forecast(steps=h)
yhat.index = valid.get_dates()
print(yhat.head())
```