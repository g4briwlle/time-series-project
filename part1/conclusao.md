# Comparação dos modelos e conclusão

A avaliação foi realizada nos 28 dias de validação, de 28/03/2016 a
24/04/2016. Comparamos o SARIMA com quatro baselines: média, naive,
naive sazonal semanal (m = 7) e drift.

As métricas utilizadas foram MAE, RMSE e MASE. Para cada série, o MASE
utilizou a mesma escala: o MAE in-sample do naive sazonal semanal
calculado exclusivamente no treino. Em todas as métricas, valores
menores indicam melhor desempenho.

## Resultados

| Série | Modelo | MAE | RMSE | MASE |
|---|---|---:|---:|---:|
| FOODS | Média | 526,93 | 671,83 | 1,540 |
| FOODS | Naive | 664,64 | 722,86 | 1,943 |
| FOODS | Naive sazonal | 343,86 | 420,12 | 1,005 |
| FOODS | Drift | 664,97 | 723,26 | 1,944 |
| FOODS | **SARIMA** | **180,02** | **225,19** | **0,526** |
| HOBBIES | Média | 110,99 | 141,31 | 1,216 |
| HOBBIES | Naive | 169,04 | 204,73 | 1,851 |
| HOBBIES | Naive sazonal | 102,25 | 146,65 | 1,120 |
| HOBBIES | Drift | 170,30 | 205,90 | 1,865 |
| HOBBIES | **SARIMA** | **78,47** | **95,08** | **0,859** |
| store_total | Média | 813,89 | 1109,59 | 1,871 |
| store_total | Naive | 862,75 | 966,63 | 1,983 |
| store_total | Naive sazonal | 459,18 | 655,47 | 1,056 |
| store_total | Drift | 863,28 | 966,76 | 1,984 |
| store_total | **SARIMA** | **271,76** | **359,02** | **0,625** |

## Conclusão

O SARIMA apresentou os menores valores de MAE, RMSE e MASE nas três
séries, superando os quatro baselines no horizonte avaliado.

Entre os baselines, o naive sazonal semanal obteve o menor MAE e MASE
em todas as séries. Esse resultado é coerente com a sazonalidade
semanal observada no diagnóstico. Em HOBBIES, entretanto, a média
apresentou o menor RMSE entre os baselines: 141,31, contra 146,65 do
naive sazonal. Mesmo nesse caso, o SARIMA foi superior, com RMSE de 95,08.

Em comparação com o naive sazonal, o SARIMA reduziu o MAE em
aproximadamente 47,6% em FOODS, 23,3% em HOBBIES e 40,8% em store_total.
Portanto, a maior complexidade do modelo trouxe ganhos nas três séries,
com a maior redução relativa do MAE em FOODS.

O MASE do SARIMA ficou abaixo de 1 nas três séries. Isso significa que
seu MAE na validação foi menor que a escala de referência do naive
sazonal semanal calculada no treino de cada série.

Os resultados favorecem o SARIMA para este horizonte de validação.
O desempenho no holdout não foi avaliado nesta etapa.