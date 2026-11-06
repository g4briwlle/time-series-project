# Conclusão de comparação dos modelos (Task 2, entrega de 01/11)

## Como as métricas foram calculadas

A avaliação usa os 28 dias de validação (28/03/2016 a 24/04/2016), com origem em 27/03/2016. Os modelos são ajustados só com o treino e as métricas são calculadas só nesse horizonte. MAE, RMSE e MASE são recalculados pelo `run.py` a partir do `yhat` de cada modelo, sem copiar os valores dos arquivos de quem ajustou cada série.

A escala do MASE é o MAE in-sample do naive sazonal semanal (período 7) no treino, a mesma da Task 1: 435,02 em store_total, 342,09 em FOODS e 91,31 em HOBBIES. Valores menores indicam melhor desempenho, e MASE abaixo de 1 indica erro menor que o do naive sazonal in-sample.

As linhas da Task 1 (quatro baselines e SARIMA) entram como foram entregues, e o `run.py` confere que as métricas delas não mudaram. No HOBBIES entra a estratégia direta com o nome `linear`; a recursiva foi o teste da outra estratégia e não entra em `metricas.csv`.

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
| HOBBIES | Linear | 78,02 | 91,43 | 0,854 |
| store_total | Média | 813,89 | 1109,59 | 1,871 |
| store_total | Naive | 862,75 | 966,63 | 1,983 |
| store_total | Naive sazonal | 459,18 | 655,47 | 1,056 |
| store_total | Drift | 863,28 | 966,76 | 1,984 |
| store_total | SARIMA | 271,76 | 359,02 | 0,625 |
| store_total | Linear | 198,89 | 271,09 | 0,457 |

## Linear, naive sazonal e SARIMA

- **FOODS:** o linear ficou melhor que o naive sazonal e que o SARIMA nas três métricas. MAE de 146,62 contra 180,02 do SARIMA e 343,86 do naive sazonal; MASE de 0,429 contra 0,526 e 1,005.
- **HOBBIES:** o linear (estratégia direta) também ficou melhor que os dois nas três métricas, mas a diferença para o SARIMA é pequena: MAE de 78,02 contra 78,47 e MASE de 0,854 contra 0,859. Contra o naive sazonal a diferença é grande (MAE de 102,25).
- **store_total:** o linear (mínimos quadrados, estratégia direta) ficou melhor que o naive sazonal e que o SARIMA nas três métricas. MAE de 198,89 contra 271,76 do SARIMA e 459,18 do naive sazonal; MASE de 0,457 contra 0,625 e 1,056.
- Nenhum linear entregue perdeu para o naive sazonal em alguma métrica, então não há série e métrica a citar.

## Resíduos na validação

Os resíduos são y - yhat: positivo quer dizer que o modelo previu abaixo do real. As figuras estão em `part2/figures/residuos_validacao.png` (resíduos por dia) e `part2/figures/metricas_por_modelo.png` (MAE, RMSE e MASE por modelo).

- **FOODS:** o SARIMA previu abaixo do real em 20 dos 28 dias (viés médio de +107,7) e o linear previu acima em 19 dos 28 (viés médio de -87,2). Os dois erram em sentidos opostos.
- **HOBBIES:** o viés médio do linear é de +12,5, perto de zero (previu abaixo do real em 15 dias e acima em 13).
- **store_total:** o SARIMA previu abaixo do real em 21 dos 28 dias (viés médio de +188,7). O linear previu abaixo em 17 dos 28 dias e o viés caiu para +74,3.
- Os resíduos de treino de cada modelo linear: [tbd] (cada autor).

## Texto de cada série

### Série FOODS (Ximena)

[tbd]

### Série HOBBIES (Gabrielle)

#### Métricas e regularização
Para o modelo linear, usou-se uma regressão regularizada. Os hiperparâmetros de regularização ($\alpha$ e l1_ratio) foram escolhidos rigorosamente fora da validação, utilizando validação cruzada temporal (TimeSeriesSplit de 5 folds) restrita ao conjunto de treino. Inicialmente, eu imaginei que a estratégia recursiva pudesse ser melhor, já que imaginei que lags curtos pudessem influenciar o preço de um dia, mas por segurança testei o direto também e ele acabou performando melhor, provavelmente a regularização não bastou para impedir o erro acumulado das previsões passadas também usadas como input do modelo recursivo.

MÉTRICAS:

- Média: 

MAE 110,99 | RMSE 141,31 | MASE 1,216  

- Naive sazonal: 

MAE 102,25 | RMSE 146,65 | MASE 1,120  

- Linear Recursivo (DESCARTADO):

MAE 81,01 | RMSE 101,58 | MASE 0,887

- SARIMA: 

MAE 78,47 | RMSE 95,08 | MASE 0,859

- Linear Direto (escolhido como o modelo linear): 

MAE 78.01 | RMSE 91.42 | MASE 0.854

O linear de estratégia recursiva ultrapassou o de média e naive sazonal, ficando um pouco atrás do SARIMA da task1. O linear de estratégia direta ultrapassou todos e ficou com as melhores métricas. O único motivo de se escolher o direto foi a melhor performance. Os seus melhores hiperparâmetro de regressão foram: alpha=0.01, l1_ratio=1.0, mostrando que ficou a Lasso pura. 

O motivo da escolha de usar regularização no modelo direto é que usei vários lags: lag_28, lag_35, lag_42, ..., que costumam ser correlacionados (a Ridge melhora isso), e talvez defasagens muito distantes (como lag_49 ou lag_56) sejam apenas ruído, bem como tem covariáveis pouco úteis (que são descartadas pela Lasso), como possivelmente o snap_CA que por lógica não afeta os valores da série de hobbies por ser relacionado com comida. Além, claro, de minimizar o overfitting.

#### Covariáveis:

A construção das covariáveis seguiu o rigor de não gerar vazamento de dados, utilizando apenas informações de calendário e o histórico disponível até a origem em 27/03/2016.

No fim usei todas as covariáveis disponíveis, delegando ao próprio modelo a tarefa de seleção de atributos. Como utilizamos a regressão regularizada com penalidade L1 (Lasso), o algoritmo zera matematicamente os coeficientes das variáveis irrelevantes ou redundantes, descartando-as automaticamente da previsão final sem a necessidade de cortes manuais prévios.

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
