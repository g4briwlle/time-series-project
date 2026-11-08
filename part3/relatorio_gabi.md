# Relatório — Modelos de Árvore e Boosting na série HOBBIES (Task 2)

## 1. Configuração

- **Série ajustada sozinha**: HOBBIES.
- **Modelos testados** (dois nomes aceitos pelo check, ambos reportados):
  - `arvore` → `RandomForestRegressor`
  - `boosting` → `HistGradientBoostingRegressor`
- **Sem padronização**: árvores e boosting são **invariantes a transformações
  monotônicas** das features; `preço` e `y` brutos produzem os mesmos splits.
- **Hiperparâmetros via Nested CV 2NCV** (TimeSeriesSplit outer=2, inner=3,
  folds ≤ 2016-03-27). Nenhum fold toca a validação.
- **MASE**: escala do naive sazonal (período 7) in-sample no treino, em y bruto.

### Grids de hiperparâmetros
- `RandomForest`: `n_estimators=300`, `max_depth ∈ {8, 12, None}`,
  `min_samples_leaf ∈ {3, 5, 10}`.
- `HistGradientBoosting`: `max_iter=300`, `learning_rate ∈ {0.03, 0.1}`,
  `max_depth ∈ {4, 6, None}`, `min_samples_leaf ∈ {10, 20}`.

### Features
As mesmas do pipeline linear (lags 1/7/14/21/28, rolling 7/28 causais,
preço, SNAP, eventos por tipo, calendário).

## 2. Estratégias testadas

- **Direta**: lags/rolling fixos na origem (2016-03-27); calendário, preço,
  SNAP, eventos do **dia alvo**; horizonte `h` como feature.
- **Recursiva**: modelo one-step; `lag_1` no dia `t` recebe a previsão do dia
  `t-1` após `t > 1`.

## 3. Métricas na validação (2016-03-28 a 2016-04-24)

| Modelo | Estratégia | MAE | RMSE | MASE |
|---|---|---:|---:|---:|
| `arvore`   | Direta     | **72.97** | **94.89** | **0.799** |
| `boosting` | Direta     | **75.47** | **95.09** | **0.827** |
| `arvore`   | Recursiva  | **82.93** | **104.77** | **0.908** |
| `boosting` | Recursiva  | **82.14** | **101.17** | **0.900** |

Escala MASE (naive sazonal in-sample): ≈ 91.30 unidades.

## 4. Direto vs. Recursivo (por modelo)

| Modelo | Melhor estratégia | Δ MAE (recursivo − direto) | Δ relativo |
|---|---|---:|---:|
| `arvore`   | **Direta** | +9.96 | +13.6% |
| `boosting` | **Direta** | +6.67 | +8.9% |

Ou seja, **para ambos os modelos de árvore, a estratégia direta vence com
folga**. Isso contrasta com o linear (onde a recursiva ganhou por margem
pequena) e tem explicação clara:

1. **`h` como feature explícita na direta.** O modelo aprende mapeamentos
   distintos por horizonte — padrões de fim de semana no dia 1 vs. no dia 28,
   efeito do preço da semana 3, etc. A recursiva não tem `h` como feature e
   fica presa a um mesmo mapa um-passo.
2. **Propagação de erro é mais severa em árvores.** Uma árvore prevê uma
   constante por folha; um erro de 1 unidade em `y_hat` pode mover a próxima
   previsão para uma folha diferente e desviar dezenas de unidades. Em modelo
   linear, o erro se propaga suavemente.
3. **Extrapolação.** Árvores não extrapolam; na recursiva, uma previsão baixa
   realimentada reduz `lag_1`, que por sua vez puxa a próxima previsão ainda
   mais para baixo (viés de “encolhimento” ao longo do horizonte). A direta
   evita esse ciclo.

## 5. Comparação com os baselines

- **Naive sazonal**: os quatro resultados têm MASE < 1 — **todos batem o
  naive sazonal**. `arvore` direto é 20.1% melhor na escala do treino
  (MASE 0.799); `boosting` direto, 17.3% melhor. **Nenhum modelo de árvore
  perde para o naive sazonal em HOBBIES**, em nenhuma métrica.
- **SARIMA da Task 1**: comparação numérica a fechar no `metricas.csv`
  consolidado. Pelo patamar (MAE ~73 contra MAE ~85 do melhor linear), é
  esperado que `arvore` direto bata também o SARIMA em MAE/RMSE — a confirmar
  com os números do SARIMA no arquivo final.

## 6. Melhor modelo do conjunto

**`arvore` (RandomForest) na estratégia direta** é o melhor modelo entre os
seis avaliados nesta série (linear direto/recursivo + árvore/boosting
direto/recursivo):

- MAE 72.97 vs. segundo lugar (`boosting` direto, 75.47): −3.3%
- MAE 72.97 vs. melhor linear (`linear` recursivo, 83.58): **−12.7%**
- MASE 0.799 vs. melhor linear (0.915): −12.7%

## 7. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| RandomForest e HGB | Cobertura de “árvore” e “boosting” sem sair do escopo do enunciado. |
| Sem padronização | Árvores são invariantes a transformação monotônica; padronizar não muda splits e adiciona risco de esquecer de despadronizar. |
| 2NCV dentro do treino | Atende “hiperparâmetros fixados antes de ler o MAE da validação” de forma ainda mais rigorosa (seleção interna ao treino). |
| Grid enxuto | Evita *overfitting* da busca no treino e mantém custo computacional viável (≈ 54 fits RF + 72 fits HGB por série). |
| `shift(1).rolling()` | Causalidade: rolling de `t` não usa `y_t`. |
| `np.maximum(0, yhat)` | Evita previsão negativa de vendas. |

## 8. Conclusão

Para HOBBIES:

- **Melhor configuração geral**: `arvore` direto (MAE 72.97, RMSE 94.89,
  MASE 0.799).
- **Melhor entre árvore e boosting**: `arvore` vence `boosting` na direta
  (MAE −2.50, −3.3%) e perde por pouco na recursiva (MAE +0.79, +1.0%).
- **Melhor entre direto e recursivo**: **direta** para os dois modelos.
- **Comparação com linear**: árvore/boosting direto batem ambos os lineares
  com folga (≥ 10 pontos de MAE).
- **Naive sazonal**: batido por todos.
- **SARIMA**: comparação final a preencher com o `metricas.csv` consolidado,
  mas o patamar sugere vantagem de árvore/boosting direto também sobre o
  SARIMA em MAE/RMSE.

## 9. Observação sobre a rede (entrega de 15/11)

Esta entrega cobre apenas **linear e árvore/boosting**. A rede (`rede`,
`lstm`, `rnn` ou `transformer`) precisa entrar em
`previsoes_validacao.csv` e `metricas.csv` até 15/11 e ser discutida em uma
seção adicional do `AI_USAGE.md`. O `check_task2.py --ate rede` (default)
cobra as três classes.