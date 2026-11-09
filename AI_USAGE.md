## Ferramentas utilizadas

- Claude Code (Anthropic): baselines e revisão do PR de diagnóstico.

## Onde ajudou

- **Baselines:** rascunho do código dos quatro baselines (`src/baselines.py`) e do erro in-sample do naive sazonal, que serve de escala do MASE; checagem das previsões contra as fórmulas da aula e contra o `check_task1.py`.
- **Revisão do PR de diagnóstico:** conferência das figuras, do split e do `AI_USAGE.md` contra o `check_task1.py` antes do merge.

## Exemplos de prompts

- **Baselines:** "Draft the code for the mean baseline."
- **Revisão do PR de diagnóstico:** "Help me review the open PR from my colleague. Is there anything wrong? I couldn't see any issue in my own analysis."

## Erros corrigidos

- **Baselines:** a IA fez a função de previsão receber o `DataReader` da validação inteiro, com o y, quando ela só precisava das datas. A revisão do PR pegou isso, e a função passou a receber só `valid.get_dates()`; assim, o código dos baselines não tem acesso ao y da validação. A mesma revisão apontou um comentário fora de lugar no topo do arquivo, que removemos.

## Responsabilidade

A equipe se responsabiliza pelo conteúdo entregue neste trabalho.

A IA ajudou a escrever código, mas o papel humano foi essencial: definimos o escopo de cada tarefa, orientamos a ferramenta ao longo do trabalho e só levamos código para o repositório depois de lê-lo, rodá-lo e conferir os resultados.

---

# Task 2: linear (entrega de 01/11)

## Ferramentas utilizadas

- Claude Code (Anthropic): organização dos materiais da Task 2 e código de integração (`run.py`).
- Gemini (Google): enumeração das tarefas da Task 2 e duas propostas de divisão do trabalho no grupo.
- DeepSeek: explicação do README da Task 2 para o grupo, estudo da modelagem e código do modelo linear da série store_total.
- Assistente de IA em chat: estudo do enunciado e código do modelo linear da série HOBBIES.

## Onde ajudou

- **Organização dos materiais:** leitura do enunciado e do `check_task2.py`, divisão do que cada corte (`linear`, árvore, rede) cobra e plano do que fechar antes de cada domingo. A lista das tarefas e a divisão entre as cinco pessoas partiram de uma conversa com o Gemini, e o README da Task 2 foi explicado ao grupo com o DeepSeek.
- **Integração:** a parte mecânica do `run.py`: leitura do CSV de previsões de cada série, validações (28 dias, sem datas repetidas, sem NaN), cálculo de MAE, RMSE e MASE com a escala do naive sazonal, conferência de que as métricas da Task 1 não mudaram e escrita de `previsoes_validacao.csv` e `metricas.csv`. Também o passo a passo do `README.md`.
- **Modelo linear de HOBBIES:** estruturação das features (calendário, eventos, preço padronizado só com o treino), lags e escolha de hiperparâmetros por validação cruzada temporal dentro do treino, nas estratégias recursiva e direta.
- **Modelo linear de store_total:** conceitos (viés e variância, estratégia direta contra recursiva em 28 passos, por que o menor lag seguro é 28), a decisão de não usar regularização (mínimos quadrados, com cerca de 130 observações por coluna), a estrutura das features (dummies de dia da semana, SNAP, eventos, preço e alvo padronizados só com o treino) e, ao final, o código do pipeline.

## Exemplos de prompts

- **Estudo do enunciado (HOBBIES):** "a partir do enunciado desse trabalho de series temporais, que é a continuação da parte1 já feita, me ajude a entender como vou fazer as coisas. tambem me explique os conceitos."
- **Estratégia direta (HOBBIES):** "eu poderia fazer uma implementação recursiva e uma direta pra ver a diferença?" e, em seguida, "faça uma implementação de uma estrategia direta em outro modulo model_direto_hobbies".
- **Estratégia (store_total):** "Devo fazer a regressão linear direta ou recursiva para `store_total` em 28 passos? Justifique." e "Por que usar `lag-28` e não `lag-364` (um ano) para `store_total`?"
- **Regularização (store_total):** "Devo usar regularização no `linear`? O argumento de ML a favor parece válido."
- **Features sem vazamento (store_total):** "Me mostre como construir as features (dummies de dia da semana, SNAP, eventos, preço, `lag-28`) sem vazamento, com `date` como coluna."
- **Organização (integração):** "Plan everything I need to do." com o enunciado, o README do check e a divisão de tarefas anexados.

## Erros corrigidos

- **Fonte das previsões da Task 1:** a primeira versão do `run.py` lia os baselines e o SARIMA de `part1/previsoes_validacao.csv`. A conferência contra `part1/metricas.csv` mostrou que o SARIMA desse arquivo vinha de outro ajuste (diferença de até 0,04 no yhat), e as métricas da Task 1 não fechavam. A fonte passou a ser `part1/outputs`, que reproduz as métricas entregues com diferença da ordem de 1e-12.
- **Caminhos do modelo de HOBBIES:** o script lia `dados/` e gravava os CSVs relativos ao diretório de execução, então falhava quando rodado da raiz pelo `run.py`. Os caminhos passaram a ser relativos ao arquivo.
- **Vazamento no lag de store_total:** a primeira versão calculava o lag de 28 dias sobre uma base sem índice de datas e podia puxar o y da validação. O lag passou a receber só o y padronizado do treino, exige `DatetimeIndex`, e a base de validação confere com um `assert` que toda data usada no lag é de até 27/03/2016.
- **Caminho dos dados de FOODS:** o script lia `part2/projeto-task2/dados`, pasta que não existe depois da reorganização. O caminho passou a apontar para `part2/dados`, a mesma base dos outros modelos.

## Responsabilidade

A equipe se responsabiliza pelo conteúdo entregue nesta Task. A IA ajudou a organizar os materiais e a escrever a parte mecânica do código, e só levamos código para o repositório depois de lê-lo, rodá-lo e conferir os resultados contra o `check_task2.py`.

---

# Task 3: árvore e boosting (entrega de 08/11)

Esta seção cobre só a integração da entrega (`run.py`, `README.md` e o check). Os itens de cada modelo de árvore ou boosting (HOBBIES, store_total e FOODS) são acrescentados por quem o escreveu.

## Ferramentas utilizadas

- Claude Code (Anthropic), como apoio na integração: leitura das branches e rascunho de trechos do `run.py` e do README, que a equipe revisou.

## Onde ajudou

- **Leitura das branches:** levantamento do que cada branch da Task 3 entrega por série (HOBBIES com `arvore` e `boosting`, FOODS só com a rede, store_total sem saídas ainda). Esse levantamento serviu de base para a equipe definir o escopo.
- **Rascunho do `run.py`:** proposta de leitura dos CSVs com mais de um nome de modelo por arquivo, conferência do nome e aviso das séries sem `linear` ou sem `boosting`/`arvore`.
- **README:** rascunho da tabela de arquivos e scripts.
- **Check:** execução do `check_task2.py --ate arvore` para listar o que ainda falta.

## O que a equipe decidiu

- O escopo: integrar só o que já existe, sem ajustar os modelos que faltam em FOODS e store_total.
- O repositório oficial da entrega (`g4briwlle/time-series-project`) e a base da branch.
- A estratégia direta nas séries em que há duas (HOBBIES), seguindo a Task 2.
- A ordem: PRs dos colegas primeiro, consolidação depois.
- O que entra no README e nesta seção, depois de revisar o PR.

## Exemplos de prompts

- **Escopo:** "This is for the time series work. Found the folder it is in." com a conversa do grupo anexada.
- **Entrega:** "You should merge ximana/daniel prs and open a pr consolidating."

## Erros corrigidos

- **Refit no `run.py`:** a leitura da nova estrutura mostrou que o `run.py` ainda chamava `part2.src.model_store_total.model_store_total`, módulo que virou `models/linear.py`. Os scripts de HOBBIES também voltaram a ler `dados/` e a gravar os CSVs no diretório de execução, onde o `run.py` não os procura. O `run.py` passou a só consolidar os CSVs versionados, e o README diz isso e lista o script de cada arquivo.
- **Rótulo no store_total (apontado ao autor, ainda aberto):** `save_outputs` grava `modelo="linear"` também nas saídas de árvore e boosting. O `run.py` recusa o arquivo se o nome do modelo não for `arvore` ou `boosting`, para não duplicar as linhas do linear.

## Responsabilidade

A equipe se responsabiliza pelo conteúdo entregue nesta Task. A IA foi usada como apoio; o escopo e as decisões foram da equipe, e só levamos código para o repositório depois de lê-lo, rodá-lo e conferir os resultados contra o `check_task2.py`.
