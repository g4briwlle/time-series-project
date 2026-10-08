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

# Rodando arquivo `run.py`

Para rodar o arquivo `run.py` e gerar tanto `metricas.csv` quanto `previsoes_validacao.csv` execute (com o ambiente virtual criado e ativado, preferencialmente):

```bash
pip install uv && uv pip install -r requirements.txt && python run.py
```

Obs.: Utiliza-se uv para instalação drasticamente mais rápida das dependências.
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

## Resultados

A comparação dos modelos está em [conclusao.md](conclusao.md).
