## Classe rede — `store_total`

### Modelo e estratégia

MLP em scikit-learn (`MLPRegressor`), com um ajuste por série. Para `store_total`, foram implementadas as duas estratégias — **direta** (um modelo por horizonte) e **recursiva** (um modelo one-step realimentado) — e a escolha entre elas foi feita pelo erro médio na validação cruzada temporal interna, restrita ao treino até 2016-03-27. A recursiva venceu (CV MASE = 0,7399) e é a que entra em `metricas.csv`. Na recursiva, a previsão de cada dia retorna como lag do dia seguinte; em nenhum passo o y observado no horizonte de validação entra como feature — o que retorna é sempre ŷ do próprio modelo.

### Features

`build_exog_features` constrói as entradas a partir de: dummies de dia da semana; `snap_CA`; um indicador binário `is_event` e dummies de `event_type_1` e `event_type_2`; `preco_store_total` padronizado; e três janelas móveis calculadas sobre a série-alvo — `rolling_mean_7`, `rolling_mean_28` e `rolling_std_7`.

Ficaram de fora `event_name_1`, `event_name_2`, `preco_FOODS` e `preco_HOBBIES`: os nomes de evento são de alta cardinalidade e esparsos, e os preços por categoria não têm relação direta com `store_total` (que já é capturado por `preco_store_total`). A série-alvo e `preco_store_total` foram padronizados com média e desvio calculados somente no treino; a mesma transformação foi aplicada ao horizonte de validação. As janelas móveis na recursiva são construídas com dados até 2016-03-27 na origem e, depois, com as previsões do próprio modelo — nunca com y observado.

### Lags

`[1, 2, 3, 4, 5, 6, 7, 14, 21, 28]` — cobre um ciclo semanal completo, dois ciclos e um mensal.

### Grade e validação cruzada

A busca foi exaustiva sobre `hidden_layer_sizes` ∈ `{(32,), (64,)}` × `alpha` ∈ `{1e-4, 1e-3, 1e-2, 1e-1}` × `learning_rate_init` ∈ `{1e-4, 1e-3, 1e-2}`, totalizando 24 combinações.

A validação cruzada usa folds expansivos próprios (`make_cv_folds`): 4 dobras de 28 dias cada, cobrindo os últimos 112 dias do treino (2015-12-07 a 2016-03-27). Cada dobra treina em tudo que antecede sua janela e valida nos 28 dias seguintes; a métrica de seleção é o MASE médio nas 4 dobras. Nenhuma dobra toca o horizonte de validação publicado (2016-03-28 a 2016-04-24).

### Critério de parada

`early_stopping=False` foi mantido para que o `MLPRegressor` não crie nenhuma fatia interna de validação. O treino para por `max_iter=1000` (teto) ou quando a perda de treino não melhora mais que `tol=1e-4` por `n_iter_no_change=10` épocas consecutivas — ambos critérios sobre o próprio treino, sem contato com o horizonte publicado.

### Resultado (validação, 28 dias)

MAE 221,89, RMSE 291,66, MASE 0,510.

| Modelo | MAE | RMSE | MASE |
|---|---:|---:|---:|
| naive_sazonal | 459,18 | 655,47 | 1,056 |
| sarima | 271,76 | 359,02 | 0,625 |
| **rede (MLP recursivo)** | **221,89** | **291,66** | **0,510** |

### Comparação

A rede supera o naive sazonal nas três métricas (−51,7% em MAE, −55,5% em RMSE, −51,7% em MASE) e supera também o SARIMA nas três (−18,4% em MAE, −18,8% em RMSE, −18,4% em MASE). Não há série ou métrica em que a rede perca para o naive sazonal em `store_total`.

### Forma do erro

A razão RMSE/MAE é 1,31 para a rede, contra 1,43 no naive sazonal e 1,32 no SARIMA: a distribuição de erros da rede é parecida com a do SARIMA, com concentração moderada — há dias com erro acima da média, mas sem cauda extrema dominante.

### CV vs. validação

O MASE de validação cruzada interna (0,740) é pior que o MASE no horizonte publicado (0,510). As 4 dobras avaliam os últimos 112 dias do treino (2015-12-07 a 2016-03-27); o início de 2011 entra apenas como treino, não como avaliação. A distância entre os dois números tem duas origens identificáveis. Primeiro, a dobra 0 contém o Natal de 2015 (25/12, y=0), em que o modelo erra 2.319 unidades; sem esse dia, a dobra 0 vai de 0,994 para 0,833, e a média do CV cai de 0,740 para 0,700. Segundo, a janela de validação (final de março a abril de 2016) é intrinsecamente mais fácil que as janelas de dezembro a março cobertas pelas dobras. O que importa para o critério de split é que o modelo não foi escolhido pelo erro na validação publicada: a seleção de hiperparâmetros aconteceu inteiramente dentro do treino, e a validação foi tocada uma única vez.

### Caveat da recursiva com lag 28

No passo 28, o lag 28 ainda é o y observado do último dia de treino (2016-03-27); os demais lags (1 a 7, 14 e 21) e as janelas móveis já são previsões do próprio modelo. Ou seja, a parte mais "antiga" do contexto ainda é real, mas a maior parte do vetor de entrada já é sintética. Esse é o custo conhecido da recursiva, e é justamente por isso que a direta também foi implementada e comparada no CV interno; a recursiva venceu nesta série, o que é uma propriedade dos dados, não uma garantia geral.

### Comparação com as demais classes

Em `store_total`, a `rede` (MAE 221,89) fica atrás do `linear` (198,89) e do `boosting` (173,06). A rede supera os dois baselines da Task 1, mas não é a classe de melhor desempenho nesta série. O resultado é consistente com a literatura: em dados tabulares de tamanho moderado (~1.900 pontos), métodos de árvore com boosting costumam ser competitivos com redes neurais, e a rede foi mantida porque o critério da Task 4 exige a classe e porque a comparação entre as três classes é parte do que se pede.

# AI_USAGE

Ferramentas usadas (Item 1): Python 3.x com scikit-learn (MLP), pandas/numpy para manipulação e joblib para paralelização do grid search. Usamos DeepSeek como apoio de redação, revisão de código e comparação de arquiteturas; nenhuma previsão ou métrica foi gerada diretamente pela IA sem execução local e verificação contra o código.

Onde ajudou (item 2): features (codificação de eventos, escolha dos lags múltiplos de 7), rede (comparação MLP × LSTM × Transformer e decisão por eficiência amostral).

Prompts representativos (item 3):

"Compare MLP, LSTM e Transformer para vendas diárias com ~1.900 pontos de treino e horizonte de 28 dias; qual é defensável e por quê?" — A resposta sugeriu Transformer; o grupo descartou com base na literatura de eficiência amostral e manteve MLP/LSTM.

```md
I think the dataframe has bot the index and a column as "dates":

```bash
Traceback (most recent call last):
  File "<frozen runpy>", line 203, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/daniel/desktop/graduacao/series-temporais/projeto/time-series-project/part2/src/model_store_total/models/mlp/cross_validation.py", line 220, in <module>
    cv_out = grid_search_cv(train_df, cov_df, SERIE,horizon=28, n_folds=4, gap=0, n_jobs=-1)
  File "/home/daniel/desktop/graduacao/series-temporais/projeto/time-series-project/part2/src/model_store_total/models/mlp/cross_validation.py", line 139, in grid_search_cv
    folds = make_cv_folds(train_df, horizon=horizon, n_folds=n_folds, gap=gap)
  File "/home/daniel/desktop/graduacao/series-temporais/projeto/time-series-project/part2/src/model_store_total/models/mlp/cross_validation.py", line 42, in make_cv_folds
    df = train_df.sort_values("date").reset_index(drop=True)
         ~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/home/daniel/desktop/graduacao/series-temporais/projeto/time-series-project/.venv/lib/python3.14/site-packages/pandas/core/frame.py", line 9330, in sort_values
    k = self._get_label_or_level_values(by[0], axis=axis)
  File "/home/daniel/desktop/graduacao/series-temporais/projeto/time-series-project/.venv/lib/python3.14/site-packages/pandas/core/generic.py", line 1701, in _get_label_or_level_values
    self._check_label_or_level_ambiguity(key, axis=axis)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/home/daniel/desktop/graduacao/series-temporais/projeto/time-series-project/.venv/lib/python3.14/site-packages/pandas/core/generic.py", line 1661, in _check_label_or_level_ambiguity
    raise ValueError(msg)
ValueError: 'date' is both an index level and a column label, which is ambiguous.
```
```
- O modelo sugeriu uma normalização nas funções que manipulam os dados que corrigia a ambiguidade gerada

Erro corrigido (item 4):

"A primeira versão padronizava série e preços com média e desvio de treino + validação concatenados. Corrigido para treino apenas, antes de gerar yhat."


Responsabilidade (Item 5): O grupo assume responsabilidade integral pelo código, pelas previsões e pelas métricas entregues. Todas as sugestões da IA foram revisadas, executadas e validadas contra o código antes de entrar no relatório. Os erros listados no item 4 foram identificados e corrigidos pelo grupo. Nenhuma métrica foi copiada de resposta de IA sem reprodução local.