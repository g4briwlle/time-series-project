# Rede Neural na série FOODS (Task 2)

## 1. Configuração

- **Série ajustada sozinha**: FOODS (loja Walmart CA_1).
- **Modelo testado**: `rede` / `lstm` → LSTM com cabeças densas por horizonte.
- **Janela de dados**: treino de 29/01/2011 a 27/03/2016 (1.885 observações); validação oficial de 28/03/2016 a 24/04/2016 (28 dias).
- **Pré-processamento**: padronização de vendas e preços; codificação de indicadores categóricos (one-hot/dummy). Scaler e encoders ajustados estritamente dentro do treino de cada corte.
- **Seleção interna**: validação temporal expansiva em 3 blocos internos de 28 dias (origens em 03/01, 31/01 e 28/02 de 2016), minimizando o MAE médio na escala original.
- **Otimização**: semente 42, Adam com taxa de aprendizado 0,001, batches temporais de até 128 exemplos, função de perda MAE (L1) nas vendas padronizadas.
- **MASE**: escala do naive sazonal semanal in-sample no treino ($342{,}09$ unidades).

## 2. Estratégia de previsão

- A rede gera as previsões dos 28 dias simultaneamente a partir da origem (27/03/2016).
- Nenhuma previsão $\hat{y}_t$ entra como input de $\hat{y}_{t+1}$, eliminando o acúmulo de erros de predição ao longo do horizonte.
- Despadronização para a escala real e corte inferior via $\max(0, \hat{y})$ para eliminar previsões de vendas negativas.

## 3. Seleção de hiperparâmetros (cortes internos)

| Histórico | Épocas | MAE interno médio |
|---:|---:|---:|
| **56 dias** | **20** | **206,75** |
| 28 dias | 20 | 222,14 |
| 28 dias | 40 | 237,50 |
| 56 dias | 40 | 245,86 |

A configuração selecionada (**56 dias de histórico e 20 épocas**) foi retreinada com todas as 1.802 amostras sobrepostas do conjunto de treino até a origem oficial.

## 4. Métricas na validação oficial (2016-03-28 a 2016-04-24)

| Modelo | MAE | RMSE | MASE |
|---|---:|---:|---:|
| **`lstm` (Rede)** | **126,59** | **155,01** | **0,370** |
| `lasso` (Linear) | 146,62 | 177,57 | 0,429 |
| `sarima` (Task 1) | 180,02 | 225,19 | 0,526 |
| `naive_semanal` | 343,86 | 420,12 | 1,005 |

Escala MASE (naive sazonal in-sample): 342,09 unidades.

## 5. Comparação com os baselines e modelos alternativos

| Modelo base | MAE do modelo | MAE da LSTM | Redução absoluta | Redução relativa |
|---|---:|---:|---:|---:|
| Naive semanal | 343,86 | **126,59** | −217,27 | **−63,2%** |
| SARIMA | 180,02 | **126,59** | −53,43 | **−29,7%** |
| Lasso | 146,62 | **126,59** | −20,03 | **−13,7%** |

- A rede reduziu o MAE em 13,7% ante o Lasso, capturando relações não lineares complexas entre o preço, calendário SNAP e a dinâmica diária da categoria FOODS.
- Superou amplamente o SARIMA (−29,7%) e o Naive sazonal (−63,2%), demonstrando o impacto crucial das variáveis exógenas no horizonte estendido.

## 6. Análise de erros e comportamento residual

- **Viés geral**: erro médio ($\text{real} - \text{previsto}$) de **+35,05 unidades**. A rede subestimou a demanda em 18 dias e superestimou em 10 dias.
- **Padrão em domingos**: o modelo apresentou subestimação sistemática nos 4 domingos da validação, com erro médio de **+190,87 unidades**.
- **Pior caso pontual**: o maior erro absoluto ocorreu no último domingo (24/04/2016), com subestimação de **306,49 unidades**.
- **Ressalva amostral**: o horizonte contém apenas 4 domingos; a concentração de erro indica um pico de demanda pontual não totalmente capturado pelo histórico de 56 dias, demandando mais origens de teste para generalização.

## 7. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| LSTM com 16 unidades | Capacidade de representação temporal suficiente para a série sem incorrer em overfitting severo. |
| Estrutura direta (multi-head) | Evita a propagação e amplificação de erros típicos de modelos autoregressivos recursivos em 28 passos. |
| Compartilhamento de pesos na saída | Reduz drasticamente o número de parâmetros livres e impõe regularização estrutural entre os horizontes. |
| Treino expansivo em 3 cortes | Garante alinhamento com os folds da validação sem vazamento de dados. |
| Janela histórica de 56 dias | Captura o ciclo completo de 8 semanas de sazonalidade e defasagens de curto prazo com menor MAE interno. |
| $\max(0, \hat{y})$ no pós-processamento | Não existem vendas negativas. |

## 8. Conclusão

Para a categoria FOODS:

- **LSTM** direta com histórico de 56 dias e 20 épocas.
- **MAE 126,59**, **RMSE 155,01** e **MASE 0,370**.
- *Lidera com folga todas as métricas frente ao Lasso, SARIMA e Naive sazonal.
- Subestimação nos picos de domingo (+190,87 unidades em média), que pode ser mitigada em iterações futuras com pesos amostrais ou features específicas de interação fim-de-semana/SNAP.
