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

Esta seção cobre a integração da entrega (`run.py`, `README.md` e o check) e o script de árvore e boosting de FOODS. Os itens de cada modelo de árvore ou boosting (HOBBIES, store_total e FOODS) são acrescentados por quem o escreveu.

## Ferramentas utilizadas

- Claude Code (Anthropic), como apoio na integração: leitura das branches e rascunho de trechos do `run.py` e do README, que a equipe revisou.

## Onde ajudou

- **Leitura das branches:** levantamento do que cada branch da Task 3 entrega por série (HOBBIES com `arvore` e `boosting`, FOODS só com a rede, store_total sem saídas ainda). Esse levantamento serviu de base para a equipe definir o escopo.
- **Rascunho do `run.py`:** proposta de leitura dos CSVs com mais de um nome de modelo por arquivo, conferência do nome e aviso das séries sem `linear` ou sem `boosting`/`arvore`.
- **README:** rascunho da tabela de arquivos e scripts.
- **Modelo de FOODS:** rascunho do script de árvore e boosting de FOODS (`part3/src/foods_model/model_direto_foods.py`), adaptado do modelo direto de HOBBIES, com hiperparâmetros fixos a priori, sem busca e sem olhar a validação. O resultado foi conferido contra o `check_task2.py` antes de entrar.
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
- **Rótulo e saídas no store_total:** o `save_outputs` gravava `modelo="linear"` também nas saídas de árvore e boosting, e o `main` da busca em grade tinha o grid da árvore comentado, comparava a estratégia com `'direct'` em vez de `'direto'` e sobrescrevia o eixo `estrategia` com uma string, o que multiplicava a grade. Foram corrigidos na branch do autor, a busca foi refeita e os CSVs saíram com `arvore` e `boosting` em `part3/outputs`. O `run.py` recusa um arquivo se o nome do modelo não for `arvore` ou `boosting`.

## Responsabilidade

A equipe se responsabiliza pelo conteúdo entregue nesta Task. A IA foi usada como apoio; o escopo e as decisões foram da equipe, e só levamos código para o repositório depois de lê-lo, rodá-lo e conferir os resultados contra o `check_task2.py`.

---

# Task 4: rede (entrega de 15/11)

Esta seção descreve o uso do Claude (Anthropic) na parte do Kauan da Task 4. Os commits em que o Claude aparece como coautor são três, todos já na main, e são os únicos detalhados abaixo. O resto da Task 4 não está aqui: as redes de FOODS, HOBBIES e store_total e os relatórios de cada autor são deles. O rascunho desta seção também foi feito com o Claude.

| Commit | PR | O que é |
|---|---|---|
| `6f0361b` | #23 | script e figuras da rede (`part4/src/plot_rede_diagnostico.py`, `part4/figures/`) |
| `9a14e80` | #22 | correção das afirmações do relatório do store_total que o código não sustenta |
| `d821a9d` | #22 | commit de merge do #22, que leva o trailer e não tem conteúdo próprio |

## Ferramentas utilizadas

- Claude Code (Anthropic), usado pelo Kauan na parte dele da Task 4.

## Onde ajudou

- **Gráficos (`6f0361b`).** Claude: escreveu `plot_rede_diagnostico.py`, adaptando o script de gráficos de árvore e boosting da Task 3, gerou as três figuras e conferiu que elas saem idênticas dos CSVs da consolidação. Kauan: pediu os gráficos (barras de métricas, previsões e resíduos), definiu que ficariam num PR separado da consolidação e mandou o PR para a revisão de um colega, que o revisou e fez o merge.
- **Revisão do relatório do store_total (`9a14e80`).** Claude: conferiu o relatório contra o código (reexecutou `recursive_mlp.py`, que reproduz o CSV com diferença máxima de 9e-13 e o MASE de CV de 0,7399) e reescreveu os trechos que o código não sustenta (features, explicação do CV, a frase sobre quando a validação é lida e o lag 28). Kauan: decidiu revisar o PR do colega, escolheu o que perguntar ao autor no comentário do PR e fez o merge.
- **Fora da tabela.** O Kauan também usou o Claude para rascunhar a integração desta entrega no PR da consolidação (`run.py`, README, `requirements.txt`), o relatório de árvore e boosting de FOODS (`part3/relatorio_kauan.md`) e a conclusão unificada (`part4/conclusao.md`). Esses commits não levam o trailer de coautoria, e o apoio foi do mesmo tipo: o Claude rascunhou, e o escopo e as decisões foram do Kauan.

## Exemplos de prompts

- **Revisão (`9a14e80`):** "Review daniels pr and ask him anything askable or merge it".
- **Gráficos (`6f0361b`), em resumo:** gerar as barras de métricas, as previsões e os resíduos da rede, no formato dos gráficos da Task 3, em um PR separado.
- **Integração (fora da tabela):** "start the pr draft for my part", com a mensagem do grupo pedindo o CSV unificado de métricas, os relatórios e as previsões unificados.

## Erros corrigidos

- **Explicação do CV no comentário de revisão (`9a14e80`):** o comentário do Claude atribuiu a diferença entre o MASE de CV (0,740) e o de validação (0,510) a uma janela de validação mais fácil. Isso não se sustentou: o naive sazonal tem MASE 1,056 na validação e 1,018 nas dobras sem o 25/12. O relatório passou a dar a faixa das dobras em vez dessa explicação.
- **Escala do HOBBIES no relatório de FOODS:** o rascunho do Claude dizia que o HOBBIES vende cerca de 500 unidades por dia. A média da validação é 538, e o texto passou a dizer cerca de 540.
- **`torch` fora do `requirements.txt`:** a LSTM de FOODS, que já está na main, importa `torch`, e o `requirements.txt` não o listava, então o README não bastava para refazer a rede. O `torch` entrou no `requirements.txt` e o README avisa que só as LSTMs o usam.

## Responsabilidade

O Kauan se responsabiliza pelos commits da tabela e pelo que o Claude rascunhou para ele nesta Task. O escopo e as decisões foram do Kauan, e só entrou no repositório o que foi lido, rodado e conferido contra o `check_task2.py`.
