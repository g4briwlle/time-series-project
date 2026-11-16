# Conclusão de comparação dos modelos (Task 4, entrega de 15/11)

> **Rascunho do PR.** A rede de HOBBIES ainda não está nas tabelas, porque a branch `part4/gabi/lstm-model` não foi mergeada. As tabelas saem de `metricas.csv` e são regeradas depois desse merge. Apagar esta nota antes de fechar.

## Como as métricas foram calculadas

A avaliação usa os 28 dias de validação (28/03/2016 a 24/04/2016), com origem em 27/03/2016. Os modelos são ajustados só com o treino e as métricas são calculadas só nesse horizonte. MAE, RMSE e MASE são recalculados pelo `run.py` a partir do `yhat` de cada modelo, sem copiar os valores dos arquivos de quem ajustou cada série. Por isso os números desta conclusão são os de `metricas.csv`, e um relatório individual pode diferir deles na segunda casa decimal.

A escala do MASE é o MAE in-sample do naive sazonal semanal (período 7) no treino, a mesma da Task 1: 435,02 em store_total, 342,09 em FOODS e 91,31 em HOBBIES. Valores menores indicam melhor desempenho, e MASE abaixo de 1 indica erro menor que o do naive sazonal in-sample.

As tabelas trazem as três classes (`linear`, `arvore` ou `boosting`, e `rede`) com o naive sazonal e o SARIMA da Task 1. Média, naive de um passo e drift continuam em `metricas.csv`. Cada tabela está ordenada pelo MASE.

## Resultados na validação

### FOODS

| Modelo | MAE | RMSE | MASE | MAE vs naive sazonal | MAE vs SARIMA |
|---|---:|---:|---:|---:|---:|
| `rede` | 126,65 | 155,06 | 0,370 | −63,2% | −29,6% |
| `linear` | 146,62 | 177,57 | 0,429 | −57,4% | −18,6% |
| `sarima` | 180,02 | 225,19 | 0,526 | −47,6% |  |
| `boosting` | 185,24 | 235,57 | 0,541 | −46,1% | +2,9% |
| `arvore` | 195,98 | 255,66 | 0,573 | −43,0% | +8,9% |
| `naive_sazonal` | 343,86 | 420,12 | 1,005 |  |  |

### HOBBIES

A rede entra aqui quando a branch da LSTM for mergeada.

| Modelo | MAE | RMSE | MASE | MAE vs naive sazonal | MAE vs SARIMA |
|---|---:|---:|---:|---:|---:|
| `arvore` | 72,97 | 94,89 | 0,799 | −28,6% | −7,0% |
| `boosting` | 75,47 | 95,09 | 0,827 | −26,2% | −3,8% |
| `sarima` | 78,47 | 95,08 | 0,859 | −23,3% |  |
| `linear` | 85,09 | 103,09 | 0,932 | −16,8% | +8,4% |
| `naive_sazonal` | 102,25 | 146,65 | 1,120 |  |  |

### store_total

| Modelo | MAE | RMSE | MASE | MAE vs naive sazonal | MAE vs SARIMA |
|---|---:|---:|---:|---:|---:|
| `boosting` | 173,06 | 225,11 | 0,398 | −62,3% | −36,3% |
| `linear` | 198,89 | 271,09 | 0,457 | −56,7% | −26,8% |
| `rede` | 221,89 | 291,66 | 0,510 | −51,7% | −18,4% |
| `sarima` | 271,76 | 359,02 | 0,625 | −40,8% |  |
| `arvore` | 309,29 | 395,36 | 0,711 | −32,6% | +13,8% |
| `naive_sazonal` | 459,18 | 655,47 | 1,056 |  |  |

## Linear, árvore, rede, naive sazonal e SARIMA

- **Naive sazonal:** nenhum dos modelos das tabelas perde para ele, em nenhuma série e em nenhuma das três métricas. A rede é a que mais se afasta dele em FOODS (MAE 63,2% menor) e o boosting em store_total (62,3% menor).
- **SARIMA:** a rede ganha dele nas duas séries em que já entrou, FOODS (MAE 29,6% menor) e store_total (18,4% menor). Perdem para o SARIMA, nas três métricas, a árvore e o boosting em FOODS (MAE 8,9% e 2,9% maior), o linear em HOBBIES (8,4% maior) e a árvore em store_total (13,8% maior). Em HOBBIES o boosting perde só no RMSE, por 0,01 (95,09 contra 95,08).
- **Melhor modelo por série:** na rede de FOODS (MASE 0,370), no boosting em store_total (0,398) e, por enquanto, na árvore em HOBBIES (0,799).
- **O que muda de série para série:** em FOODS a rede e o linear ficam à frente do SARIMA, e as duas classes de árvore ficam atrás. Em store_total o boosting fica à frente de todos, a rede (MASE 0,510) fica entre o linear e o SARIMA e a árvore única (0,711) fica atrás do SARIMA. Em HOBBIES todos os MASE ficam entre 0,80 e 0,93, mais perto do naive sazonal (1,120) do que nas outras séries.
- **Ressalva:** é uma única janela de 28 dias, com quatro repetições de cada dia da semana. Diferenças pequenas, como o boosting contra o SARIMA em FOODS (2,9% no MAE), não bastam para afirmar uma ordem.

## A rede em cada série

- **FOODS:** LSTM com cabeças densas por horizonte, que prevê os 28 dias de uma vez, com 56 dias de histórico e 20 épocas, escolhidos em três cortes internos que terminam antes da origem. Texto em [part3/relatorio_foods.md](../part3/relatorio_foods.md).
- **store_total:** MLP recursivo de uma camada oculta de 32 neurônios, escolhido pela validação cruzada dentro do treino (MASE de CV 0,7399). Texto em [part4/relatorio_daniel.md](relatorio_daniel.md).
- **HOBBIES:** LSTM recursiva, escolhida por validação cruzada com quatro cortes. Texto em `part4/relatorio_gabi.md`, que chega com a branch `part4/gabi/lstm-model`.

## Texto de cada série

| Série | Linear | Árvore e boosting | Rede |
|---|---|---|---|
| FOODS | [part2/conclusao.md](../part2/conclusao.md) | [part3/relatorio_kauan.md](../part3/relatorio_kauan.md) | [part3/relatorio_foods.md](../part3/relatorio_foods.md) |
| HOBBIES | [part2/conclusao.md](../part2/conclusao.md) | [part3/relatorio_gabi.md](../part3/relatorio_gabi.md) | `part4/relatorio_gabi.md` |
| store_total | [part2/conclusao.md](../part2/conclusao.md) | [part3/report_daniel.md](../part3/report_daniel.md) | [part4/relatorio_daniel.md](relatorio_daniel.md) |

## Organização dos arquivos

`python run.py` lê as previsões de cada modelo (tabela de arquivos e scripts no [README](../README.md)) e regrava `previsoes_validacao.csv` e `metricas.csv` na raiz. O comando do check da entrega é `python part2/projeto-task2/checks/check_task2.py --submission . --dados dados --ate rede`.
