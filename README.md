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

```bash
source .venv/bin/activate
```

### Instalando dependências do projeto

Para facilidade, vamos usar apenas o pip para instalar as dependências listadas em requirements. Rode (com o ambiente já ativado):

```bash
pip install -r requirements.txt
```

## Rodando arquivo `run.py`

Da raiz do repo, com o ambiente virtual criado e ativado, o comando abaixo instala as dependências e reproduz `previsoes_validacao.csv` e `metricas.csv` (MAE, RMSE e MASE na validação, 28 dias):

```bash
python -m pip install -r requirements.txt && python run.py
```

O `run.py`:

1. refaz os modelos da Task 2 a partir de `part2/dados` (scripts em `part2/src` e em `part2/projeto-task2/linear_foods_model`);
2. junta as previsões de cada série com as da Task 1 (quatro baselines e SARIMA, como foram entregues, em `part1/outputs`);
3. confere que cada modelo tem uma previsão por dia da validação, sem datas repetidas;
4. recalcula MAE, RMSE e MASE a partir do `yhat`. A escala do MASE é o MAE in-sample do naive sazonal (período 7) no treino;
5. confere que as métricas da Task 1 não mudaram e grava os dois arquivos na raiz.

Para só consolidar, sem refazer os modelos, use `python run.py --so-consolidar`. Se faltar a previsão de alguma série, o script avisa no final.

### De onde vem cada classe

Cada série tem o próprio ajuste. O CSV de previsões de cada classe tem as colunas `date`, `series`, `modelo`, `yhat` e os 28 dias de 2016-03-28 a 2016-04-24:

| Classe | Série | Arquivo |
|---|---|---|
| `linear` | FOODS | `part2/projeto-task2/linear_foods_model/previsoes_validacao_foods.csv` |
| `linear` | HOBBIES | `part2/outputs/previsoes_linear_direto_HOBBIES.csv` |
| `linear` | store_total | `part2/outputs/previsoes_linear_store_total.csv` |

No HOBBIES entra a estratégia direta. Os arquivos `recursivo_` em `part2/outputs` foram o teste da outra estratégia e não entram em `metricas.csv`.

### Conferência com o check da Task 2

```bash
python part2/projeto-task2/checks/check_task2.py --submission . --dados part2/dados --ate linear
```

Troque `linear` por `arvore` ou `rede` nas entregas seguintes.

## Resultados

A comparação dos modelos da Task 1 está em [part1/conclusao.md](part1/conclusao.md) e a da Task 2 em [part2/conclusao.md](part2/conclusao.md).
