# Conclusão de comparação dos modelos (Task 2, entrega de 01/11)

## Como as métricas foram calculadas

A avaliação usa os 28 dias de validação (28/03/2016 a 24/04/2016), com origem em 27/03/2016. Os modelos são ajustados só com o treino e as métricas são calculadas só nesse horizonte. MAE, RMSE e MASE são recalculados pelo `run.py` a partir do `yhat` de cada modelo, sem copiar os valores dos arquivos de quem ajustou cada série.

A escala do MASE é o MAE in-sample do naive sazonal semanal (período 7) no treino, a mesma da Task 1: 435,02 em store_total, 342,09 em FOODS e 91,31 em HOBBIES. Valores menores indicam melhor desempenho, e MASE abaixo de 1 indica erro menor que o do naive sazonal in-sample.

As linhas da Task 1 (quatro baselines e SARIMA) entram como foram entregues, e o `run.py` confere que as métricas delas não mudaram. No HOBBIES entra a estratégia recursiva com o nome `linear`; a direta foi o teste da outra estratégia e não entra em `metricas.csv`.

## Resultados na validação

| Série | Modelo | MAE | RMSE | MASE |
|---|---|---:|---:|---:|
| FOODS | Média | 526,93 | 671,83 | 1,540 |
| FOODS | Naive | 664,64 | 722,86 | 1,943 |
| FOODS | Naive sazonal | 343,86 | 420,12 | 1,005 |
| FOODS | Drift | 664,97 | 723,26 | 1,944 |
| FOODS | SARIMA | 180,02 | 225,19 | 0,526 |
| FOODS | Linear | 146,62 | 177,57 | 0,429 |
| HOBBIES | Média | 110,99 | 141,31 | 1,216 |
| HOBBIES | Naive | 169,04 | 204,73 | 1,851 |
| HOBBIES | Naive sazonal | 102,25 | 146,65 | 1,120 |
| HOBBIES | Drift | 170,30 | 205,90 | 1,865 |
| HOBBIES | SARIMA | 78,47 | 95,08 | 0,859 |
| HOBBIES | Linear | 83,58 | 101,31 | 0,915 |
| store_total | Média | 813,89 | 1109,59 | 1,871 |
| store_total | Naive | 862,75 | 966,63 | 1,983 |
| store_total | Naive sazonal | 459,18 | 655,47 | 1,056 |
| store_total | Drift | 863,28 | 966,76 | 1,984 |
| store_total | SARIMA | 271,76 | 359,02 | 0,625 |
| store_total | Linear | 198,89 | 271,09 | 0,457 |

## Linear, naive sazonal e SARIMA

- **FOODS:** o linear ficou melhor que o naive sazonal e que o SARIMA nas três métricas. MAE de 146,62 contra 180,02 do SARIMA e 343,86 do naive sazonal; MASE de 0,429 contra 0,526 e 1,005.
- **HOBBIES:** o linear (estratégia recursiva) bateu o naive sazonal nas três métricas (MAE de 83,58 contra 102,25; RMSE de 101,31 contra 146,65; MASE de 0,915 contra 1,120), mas perdeu para o SARIMA nas três (MAE de 83,58 contra 78,47; RMSE de 101,31 contra 95,08; MASE de 0,915 contra 0,859). A diferença para o SARIMA é pequena (6,5% em MAE). A estratégia direta (MAE 85,09) foi testada e descartada por ser pior que a recursiva.
- **store_total:** o linear (mínimos quadrados, estratégia direta) ficou melhor que o naive sazonal e que o SARIMA nas três métricas. MAE de 198,89 contra 271,76 do SARIMA e 459,18 do naive sazonal; MASE de 0,457 contra 0,625 e 1,056.
- Nenhum linear entregue perdeu para o naive sazonal em alguma métrica, então não há série e métrica a citar. (Observação: em HOBBIES o linear entregue perde para o SARIMA, mas não para o naive sazonal — a frase exigida pelo enunciado se refere ao naive sazonal.)

## Resíduos na validação

Os resíduos são y - yhat: positivo quer dizer que o modelo previu abaixo do real. As figuras estão em `part2/figures/residuos_validacao.png` (resíduos por dia) e `part2/figures/metricas_por_modelo.png` (MAE, RMSE e MASE por modelo).

- **FOODS:** o SARIMA previu abaixo do real em 20 dos 28 dias (viés médio de +107,7) e o linear previu acima em 19 dos 28 (viés médio de -87,2). Os dois erram em sentidos opostos.
- **HOBBIES:** [tbd — recalcular com o modelo recursivo escolhido].
- **store_total:** o SARIMA previu abaixo do real em 21 dos 28 dias (viés médio de +188,7). O linear previu abaixo em 17 dos 28 dias e o viés caiu para +74,3.
- Os resíduos de treino de cada modelo linear: [tbd] (cada autor).

## Texto de cada série

### Série FOODS (Ximena)

[tbd]

### Série HOBBIES (Gabrielle)

#### Métricas e regularização

Para o modelo linear, usei uma regressão regularizada (Ridge/ElasticNet). Os hiperparâmetros ($\alpha$ e `l1_ratio`) foram escolhidos por validação cruzada temporal aninhada (2NCV: `TimeSeriesSplit` com 2 folds externos e 3 internos), restrita ao conjunto de treino — todos os folds terminam em ou antes de 27/03/2016. Nenhum fold tocou os 28 dias de validação. Testei as duas estratégias (direta e recursiva); a recursiva foi a melhor nas três métricas, então é ela que entra em `metricas.csv`.

Inicialmente eu imaginava que a estratégia direta pudesse ser melhor, porque ela não realimenta o modelo com as próprias previsões e, portanto, não propaga erro. Na prática, a recursiva venceu: a hipótese é que, para HOBBIES, os lags curtos (1 e 7 dias) capturam a dinâmica local melhor do que os lags longos fixos na origem, e a regularização (Ridge/ElasticNet) consegue conter o erro acumulado da realimentação ao longo dos 28 passos. O direto ficou 1,51 de MAE atrás do recursivo.

MÉTRICAS:

- Média: MAE 110,99 | RMSE 141,31 | MASE 1,216
- Naive sazonal: MAE 102,25 | RMSE 146,65 | MASE 1,120
- Linear Direto (testado, descartado): MAE 85,09 | RMSE 103,09 | MASE 0,932
- Linear Recursivo (escolhido como o modelo linear): MAE 83,58 | RMSE 101,31 | MASE 0,915
- SARIMA: MAE 78,47 | RMSE 95,08 | MASE 0,859

O linear recursivo bate o naive sazonal nas três métricas (MAE 18,3% menor, RMSE 30,9% menor, MASE 0,915 contra 1,120). Contra o SARIMA da Task 1, o linear recursivo perde nas três métricas, mas a diferença é pequena (MAE 6,5% maior, RMSE 6,6% maior, MASE 0,915 contra 0,859). A escolha do recursivo sobre o direto se deve exclusivamente à melhor performance nas três métricas.

O motivo de usar regularização é que os lags (1, 7, 14, 21, 28) e as médias móveis (7 e 28 dias) são correlacionados entre si, e há covariáveis pouco úteis para HOBBIES (por exemplo, `snap_CA`, que é ligado a comida e tende a não afetar hobbies). A penalidade L1/L2 zera ou encolhe esses coeficientes automaticamente, sem corte manual, e reduz overfitting. Os melhores hiperparâmetros encontrados pelo 2NCV foram `alpha = [preencher]` e `l1_ratio = [preencher]` (ver saída do script).

#### Covariáveis

A construção das covariáveis seguiu o rigor de não gerar vazamento de dados, utilizando apenas informações de calendário e o histórico disponível até a origem em 27/03/2016.

No fim usei todas as covariáveis disponíveis — `snap_CA`, preço da semana (`preco_HOBBIES`), eventos por tipo (`Sporting`, `Cultural`, `National`, `Religious` em `event_type_1` e `event_type_2`), dia da semana, dia do ano e seno/cosseno anual — delegando ao próprio modelo a tarefa de seleção de atributos. Como a regressão é regularizada, o algoritmo encolhe ou zera os coeficientes das variáveis irrelevantes ou redundantes, descartando-as automaticamente da previsão final sem cortes manuais prévios. O preço foi padronizado com média e desvio **só do treino**; a série também foi padronizada com média e desvio **só do treino**, e as previsões foram despadronizadas antes do cálculo de MAE/RMSE/MASE.

### Série store_total (Daniel)

Para fazer a regressão linear da série store_total, parti do seguinte pressuposto: como essa série agrega FOODS e HOBBIES, ela tem magnitude maior e, no agregado, menos ruído relativo do que as categorias individuais; por isso, um modelo parcimonioso — com poucas covariáveis bem escolhidas — tende a generalizar melhor do que um modelo com muitas features correlacionadas, que só aumentariam a variância sem ganho de sinal.

Para tanto, optei por um conjunto enxuto de covariáveis conhecidas na origem (2016-03-27): dummies de dia da semana (segunda como referência), snap_CA, flag geral de evento e dummies por tipo de evento (Cultural, National, Religious, Sporting), e o preço da semana padronizado (preco_store_total). Adicionei um lag de 28: na estratégia direta, ele é o menor lag disponível para todo o horizonte h = 1..28, e ancora o nível recente no mesmo dia da semana de quatro semanas antes. Lags curtos (lag-7, lag-1) não existem para todo o horizonte na direta; e lag-364 (um ano) descartaria ~19% do treino e ancoraria no nível de um ano atrás, menos relevante para 28 dias. A recursividade foi evitada justamente para não escalar erro de previsão ao longo dos 28 passos. Por fim, adicionei os dummies de dia da semana para modelar o padrão semanal de vendas em cada dia específico, que o lag-28 sozinho não cobre em todos os passos.

Com isso, temos cerca de 13 covariáveis para ~1.856 pontos de dados no treino (após descartar as 28 primeiras linhas sem lag), razão de ~140 pontos por covariável. Nesse regime, o OLS já tem variância baixa e dispensa regularização.

**Comparação com baselines**

| Modelo | MAE | RMSE | MASE |
|--------|-----|------|------|
| naive_sazonal | 459,18 | 655,47 | 1,056 |
| sarima | 271,76 | 359,02 | 0,625 |
| linear | 198,89 | 271,09 | 0,457 |

Contra o naive_sazonal: o linear vence nas três métricas. MAE cai 56,7% (459,18 → 198,89), RMSE cai 58,6% (655,47 → 271,09) e o MASE vai de 1,056 para 0,457 — abaixo de 1, o que significa que o modelo bate a escala do naive sazonal semanal in-sample do treino, a mesma escala usada na Task 1.

Contra o sarima da Task 1 (mesma ordem, ajuste até 2016-03-27): o linear também vence. MAE 26,8% menor (271,76 → 198,89), RMSE 24,5% menor (359,02 → 271,09) e MASE de 0,625 para 0,457.

Em store_total, portanto, o linear não perde para o naive sazonal nem para o SARIMA em nenhuma das três métricas, e não há ressalva a registrar para esta série nesse quesito. O ganho sobre o naive sazonal é o esperado conceitualmente: o naive repete a última semana do treino sem ajustar por SNAP, evento, preço ou nível do lag-28; o linear faz exatamente esse ajuste, mantendo a mesma estrutura direta (sem recursão) e sem usar o y da validação em nenhuma coluna.

## Dados e features (Henzo)

[tbd]

## Organização dos arquivos

[tbd] (Henzo)
