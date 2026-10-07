# Preparando o ambiente

### Criando e ativando ambiente virtual

Dê preferência por utilizar o Python 3.14 para esse projeto. Crie um ambiente virtual rodando:

```bash
python3.14 -m venv .venv
```

E inicie o ambiente virtual rodando:

**no Windows**:

```cmd
.\.venv\Scripts\activate.bat
```

**no Linux**:

```cmd
source .venv\bin\activate
```

### Instalando dependências do projeto

Para facilidade, vamos usar apenas o pip para instalar as dependências listadas em requirements. Rode (com o ambiente já ativado):

```bash
pip install -r requirements.txt
```

# Lendo dados com Data Reader

Para garantir o cumprimento do critério de split dos dados e evitar corrompimento deles, procure nunca acessar diretamente a pasta `dados`. Ao invés, use a classe `get_data.reader.DataReader`. Abaixo, alguns exemplos de como utilizar a classe para dados de treino e validação:

### Diagnóstico (ACF/PACF) — quem cuida do item 4

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from get_data.reader import DataReader, SERIES

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

### Baseline media — quem cuida dos quatro baselines

```python
from get_data.reader import DataReader, SERIES

train = DataReader("train")
valid = DataReader("validation")

previsoes = []
for nome in SERIES:
    y_train = train.get_series(nome)
    yhat = y_train.mean()
    for d in valid.get_dates():
        previsoes.append(
            {"date": d, "series": nome, "modelo": "media", "yhat": yhat}
        )

print(previsoes[:3])
```

### ARIMA/SARIMA — quem cuida do modelo

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
from get_data.reader import DataReader

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