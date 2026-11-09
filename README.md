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

1. lê as previsões de cada modelo, que são CSVs versionados: cada autor os gera com o próprio script (tabela abaixo) e o `run.py` não refaz os ajustes;
2. junta com as da Task 1 (quatro baselines e SARIMA, como foram entregues, em `part1/outputs`);
3. confere que cada modelo tem uma previsão por dia da validação, sem datas repetidas e com o nome de modelo esperado;
4. recalcula MAE, RMSE e MASE a partir do `yhat`. A escala do MASE é o MAE in-sample do naive sazonal (período 7) no treino;
5. confere que as métricas da Task 1 não mudaram e grava os dois arquivos na raiz.

Se faltar a previsão de alguma série, o script avisa no final quais arquivos e quais séries ficaram sem `linear` ou sem `boosting`/`arvore`.

### De onde vem cada classe

Cada série tem o próprio ajuste. O CSV de previsões de cada modelo tem as colunas `date`, `series`, `modelo`, `yhat` e os 28 dias de 2016-03-28 a 2016-04-24:

| Classe | Série | Arquivo | Script |
|---|---|---|---|
| `linear` | FOODS | `part2/projeto-task2/linear_foods_model/previsoes_validacao_foods.csv` | `part2/projeto-task2/linear_foods_model/linear_foods.py` |
| `linear` | HOBBIES | `part2/outputs/previsoes_linear_direto_HOBBIES.csv` | `part2/src/model_direto_hobbies.py` |
| `linear` | store_total | `part2/outputs/previsoes_linear_store_total.csv` | `part2/src/model_store_total/models/linear.py` |
| `arvore`, `boosting` | HOBBIES | `part3/outputs/previsoes_direto_HOBBIES.csv` | `part3/src/hobbies_model/model_direto_hobbies.py` |
| `arvore` | store_total | `part2/outputs/previsoes_arvore_direto_store_total.csv` | `part2/src/model_store_total/models/tree_boost.py` |
| `boosting` | store_total | `part2/outputs/previsoes_boosting_direto_store_total.csv` | `part2/src/model_store_total/models/tree_boost.py` |
| `arvore` ou `boosting` | FOODS | `part3/outputs/previsoes_direto_FOODS.csv` | a definir |

No HOBBIES entra a estratégia direta, tanto no linear quanto em árvore e boosting. Os arquivos `recursivo_` foram o teste da outra estratégia e não entram em `metricas.csv`. A linha de FOODS e os dois arquivos de store_total da Task 3 ainda não estão no repositório; até chegarem, o `run.py` avisa e o check de 08/11 não fecha nessas séries.

### Conferência com o check

```bash
python part2/projeto-task2/checks/check_task2.py --submission . --dados dados --ate arvore
```

Troque `arvore` por `linear` ou `rede` nas outras entregas.

## Resultados

A comparação dos modelos da Task 1 está em [part1/conclusao.md](part1/conclusao.md) e a da Task 2 em [part2/conclusao.md](part2/conclusao.md). Na Task 3, o relatório de árvore e boosting em HOBBIES está em [part3/relatorio_gabi.md](part3/relatorio_gabi.md).
