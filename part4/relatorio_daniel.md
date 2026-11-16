## Classe rede — `store_total`

### Modelo e estratégia

MLP implementado em scikit-learn (`MLPRegressor`), com um ajuste por série. Para `store_total`, foram implementadas as duas estratégias — **direta** (um modelo por horizonte) e **recursiva** (um modelo one-step realimentado) — e a escolha entre elas foi feita pelo erro médio na validação cruzada temporal interna, restrita ao treino até 2016-03-27. A recursiva venceu (CV MASE = 0,7399) e é a que entra em `metricas.csv`. Na recursiva, a previsão de cada dia retorna como lag do dia seguinte; em nenhum passo o y observado no horizonte de validação entra como feature — o que retorna é sempre ŷ do próprio modelo.

### Features

Todas as colunas de `covariaveis.csv` foram utilizadas: calendário (dia da semana e demais marcadores temporais), `snap_CA`, os dois pares de evento (`event_name_1`/`event_type_1`, `event_name_2`/`event_type_2`, codificados) e os três preços (`preco_FOODS`, `preco_HOBBIES`, `preco_store_total`). Nenhuma covariável fora do CSV foi criada. Série-alvo e preços foram padronizados com média e desvio calculados somente no treino; a mesma transformação foi aplicada ao horizonte de validação.

**Lags:** `[1, 2, 3, 4, 5, 6, 7, 14, 21, 28]` — cobre um ciclo semanal completo, dois ciclos e um mensal.

### Hiperparâmetros

Fixados antes de ler o MAE da validação e selecionados por `TimeSeriesSplit` dentro do treino:

- `hidden_layer_sizes=(32,)`
- `alpha=1e-4`
- `learning_rate_init=1e-3`
- `solver=adam`
- `early_stopping=False`

A desativação do `early_stopping` foi deliberada: com ela, nenhuma fatia de validação — nem interna — é criada pelo `MLPRegressor`, e o critério de parada passa a ser apenas `max_iter`, monitorado pela curva de perda no treino. A grade de arquitetura foi limitada a `(32,)`, `(64,)` e `(32,16)` por eficiência amostral: com ~1.900 pontos de treino, redes maiores aumentam o risco de overfitting sem ganho esperado.

### Resultado (validação, 28 dias)

| Modelo | MAE | RMSE | MASE |
|---|---:|---:|---:|
| naive_sazonal | 459,18 | 655,47 | 1,056 |
| SARIMA | 271,76 | 359,02 | 0,625 |
| **rede (MLP recursivo)** | **221,89** | **291,66** | **0,510** |

### Comparação

A rede supera o naive sazonal nas três métricas (−51,7% em MAE, −55,5% em RMSE, −51,7% em MASE) e supera também o SARIMA nas três (−18,3% em MAE, −18,8% em RMSE, −18,3% em MASE). Não há, portanto, série ou métrica em que a rede perca para o naive sazonal em `store_total`. O SARIMA é o baseline mais forte da Task 1 e, ainda assim, a rede reduz o erro absoluto médio em cerca de 50 unidades por dia.

### Forma do erro

A razão RMSE/MAE é **1,31** para a rede, contra **1,43** no naive sazonal e **1,32** no SARIMA: a distribuição de erros da rede é parecida com a do SARIMA, com concentração moderada — há dias com erro acima da média, mas sem cauda extrema dominante. Isso sugere que o ganho não vem de "acertar poucos dias muito bem", e sim de erro sistematicamente menor ao longo do horizonte.

### CV vs. validação

O MASE de validação cruzada interna (**0,740**) é pior que o MASE no horizonte publicado (**0,510**). Esse é o sentido desejável da diferença: o modelo não foi escolhido por ajuste ao período de validação, e as dobras internas incluem fases mais instáveis do treino (início de 2011, rampa de maturação da loja). Se o inverso ocorresse — CV muito melhor que validação —, haveria sinal de seleção otimista.

### Caveat da recursiva com lag 28

Como o maior lag é 28 e o horizonte também é 28, nos últimos passos todos os lags de 1 a 28 já são previsões, não observações. Os passos finais do horizonte são, na prática, extrapolação do modelo sobre si mesmo. Esse é o custo conhecido da estratégia recursiva, e é justamente por isso que a direta também foi implementada e comparada no CV interno; a recursiva venceu nesta série, o que é uma propriedade dos dados, não uma garantia geral.

# AI_USAGE

Onde ajudou (item 2): features (codificação de eventos, escolha dos lags múltiplos de 7), árvore e rede (comparação MLP × LSTM × Transformer e decisão por eficiência amostral).

Prompt representativo (item 3):

"Compare MLP, LSTM e Transformer para vendas diárias com ~1.900 pontos de treino e horizonte de 28 dias; qual é defensável e por quê?" — A resposta sugeriu Transformer; o grupo descartou com base na literatura de eficiência amostral e manteve MLP/LSTM.

Erro corrigido (item 4):

"A primeira versão padronizava série e preços com média e desvio de treino + validação concatenados. Corrigido para treino apenas, antes de gerar yhat."

Se você me passar as métricas de FOODS e HOBBIES do grupo, eu fecho os dois blocos com placeholders resolvidos e escrevo a frase final de derrota/ausência de derrota.