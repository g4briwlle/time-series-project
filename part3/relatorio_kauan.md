# Relatório — Árvore e Boosting na série FOODS (entrega de 08/11)

Script: `part3/src/foods_model/model_direto_foods.py`. Saídas: `part3/outputs/previsoes_direto_FOODS.csv` e `part3/outputs/metricas_direto_FOODS.csv`.

## 1. Configuração

- **Série ajustada sozinha**: FOODS (loja Walmart CA_1). O ajuste não usa as outras duas séries.
- **Origem e janela**: origem em 2016-03-27, treino de 2011-01-29 a 2016-03-27 e validação oficial de 2016-03-28 a 2016-04-24 (28 dias). O último `y` que entra no ajuste é o de 2016-03-27.
- **Modelos**:
  - `arvore`: `RandomForestRegressor` com 300 árvores, `max_depth=8` e `min_samples_leaf=10`. Em FOODS a classe "árvore" é uma floresta aleatória, não uma árvore única.
  - `boosting`: `HistGradientBoostingRegressor` com `max_iter=300`, `learning_rate=0.1`, `max_depth=4` e `min_samples_leaf=20`.
  - As duas com `random_state=42`.
- **Hiperparâmetros**: fixados antes de ler o erro da validação, com valores da grade do HOBBIES (copiada no topo do script), sem busca em FOODS. A validação só entra no cálculo das métricas.
- **Padronização**: as árvores não precisam padronizar `y` nem o preço, então tudo entra na escala original e nenhuma estatística sai da validação.
- **MASE**: escala de 342,09 unidades, o MAE in-sample do naive sazonal (período 7) no treino.

## 2. Estratégia de previsão: direta

Cada classe é um único modelo ajustado sobre linhas (origem, horizonte `h`), com `h` de 1 a 28 como uma das features. São 51.212 linhas e 24 features:

- **da origem**: lags de 1, 7, 14, 21 e 28 dias, médias móveis de 7 e 28 dias (deslocadas em um dia) e o `y` da própria origem;
- **do dia previsto**: `h`, dia da semana, dia do ano, seno e cosseno anuais, `preco_FOODS` da semana, `snap_CA`, indicador de evento e os tipos de evento 1 e 2 (Sporting, Cultural, National, Religious).

Todas as colunas do dia previsto já se conhecem em 2016-03-27, porque vêm de `covariaveis.csv`. Nenhuma previsão volta como lag do dia seguinte. As linhas só entram no ajuste se o `y` do dia alvo está no treino, e o script corta a previsão em zero com `max(0, ŷ)`.

## 3. Métricas na validação (2016-03-28 a 2016-04-24)

| Modelo | MAE | RMSE | MASE |
|---|---:|---:|---:|
| `arvore` | 195,98 | 255,66 | 0,573 |
| `boosting` | 185,24 | 235,57 | 0,541 |

O script foi reexecutado a partir de `dados/` e reproduz os CSVs versionados (diferença máxima de 1e-12 no `yhat`).

## 4. Comparação com o naive sazonal, o SARIMA e as outras classes

| Modelo | MAE | RMSE | MASE | MAE da árvore | MAE do boosting |
|---|---:|---:|---:|---:|---:|
| `naive_sazonal` | 343,86 | 420,12 | 1,005 | −43,0% | −46,1% |
| `sarima` | 180,02 | 225,19 | 0,526 | +8,9% | +2,9% |
| `linear` | 146,62 | 177,57 | 0,429 | +33,7% | +26,3% |
| `rede` | 126,65 | 155,06 | 0,370 | +54,7% | +46,3% |

- **Naive sazonal**: as duas classes ganham nas três métricas (MAE 43% e 46% menor).
- **SARIMA**: as duas **perdem** nas três métricas. Em MAE a árvore fica 8,9% pior e o boosting 2,9% pior. O SARIMA usa só a história da série, e as árvores, com as covariáveis, não superam essa barra em FOODS.
- **Linear e rede**: as duas perdem para os dois. Na ordem do MASE: rede 0,370, linear 0,429, SARIMA 0,526, boosting 0,541, árvore 0,573 e naive sazonal 1,005.

## 5. Análise de erros e comportamento residual

Erro = real − previsto. Gráficos das três séries em `part3/figures/previsoes_arvore_boosting.png` e `part3/figures/residuos_arvore_boosting.png`, gerados por `part3/src/plot_arvore_diagnostico.py`.

- **Viés**: a árvore tem erro médio de +31,04 (subestimou em 17 dias e superestimou em 11). O boosting tem −82,49 (superestimou em 17 dias e subestimou em 11). A média real da validação é 3.023,9, a média prevista pela árvore é 2.992,8 e a do boosting é 3.106,3.
- **Primeira semana**: o MAE de cada bloco de 7 dias a partir de 28/03 é 288, 146, 169 e 182 na árvore, e 313, 178, 134 e 116 no boosting. As duas erram mais na primeira semana.
- **Piores dias**: na árvore, sexta 01/04 (superestimou em 866,7), quinta 31/03 (subestimou em 372,2) e segunda 04/04 (subestimou em 367,4). No boosting, sábado 02/04 (superestimou em 610,8), quinta 07/04 (superestimou em 441,7) e sexta 01/04 (superestimou em 432,3).
- **SNAP**: a validação tem `snap_CA = 1` de 01/04 a 10/04, e 01/04 é o pior ou um dos piores dias das duas classes. Com um único mês de validação não dá para separar o efeito do SNAP do efeito da sexta-feira.
- **Dia da semana**: as duas superestimam nas sextas (erro médio de −312,5 na árvore e −232,3 no boosting). Cada dia da semana aparece só quatro vezes, então isso é um indício, não uma conclusão.

## 6. O que mudou em relação ao HOBBIES

O modelo reaproveita a estrutura do HOBBIES (`part3/src/hobbies_model`): estratégia direta, as mesmas features e as mesmas classes. Mudaram:

- a série e o preço (`preco_FOODS`, em vez de `preco_HOBBIES`), com `snap_CA` e eventos como no HOBBIES;
- os hiperparâmetros, que em FOODS são fixos e não passam por busca;
- a estratégia: o HOBBIES comparou direta e recursiva, e aqui só a direta foi fixada, antes de ler a validação.

A escala também muda: FOODS vende cerca de 3.000 unidades por dia na validação, contra cerca de 540 do HOBBIES, então só o MASE se compara entre as séries.

## 7. Conclusão

Em FOODS, árvore e boosting ficam bem abaixo do naive sazonal, mas atrás do SARIMA, do linear e da rede, e o boosting (MASE 0,541) é melhor que a árvore (0,573). Como os hiperparâmetros não foram buscados, parte da distância para os outros modelos pode vir desse ajuste, e uma validação cruzada temporal como a de store_total (`part3/outputs/cv_store_total.csv`) é o próximo passo natural. Essa busca não foi feita nesta entrega.
