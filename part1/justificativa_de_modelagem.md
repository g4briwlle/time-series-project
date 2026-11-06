# Justificativa de Modelagem: Escolha e Ordens do Modelo

## 1. A Escolha entre ARIMA e SARIMA
O escopo do trabalho permite a utilização de modelos ARIMA ou SARIMA. Ao analisarmos os gráficos de Autocorrelação (ACF) das três séries (`store_total`, `FOODS` e `HOBBIES`), fica evidente a presença de picos muito acentuados e persistentes nos *lags* múltiplos de 7 (7, 14, 21, 28, 35). 

Um modelo ARIMA tradicional $(p, d, q)$ regular tenta modelar a dependência apenas nos *lags* imediatamente anteriores (1, 2, 3...) e não é capaz de capturar choques que se repetem semanalmente sem que inclua-se dezenas de parâmetros (o que geraria *overfitting*). Por conta dessa forte assinatura de sazonalidade semanal identificada no diagnóstico prévio, optou-se por utilizar a extensão sazonal SARIMA $(p, d, q)(P, D, Q)_m$, fixando o período sazonal em $m=7$.

---

## 2. Identificação das Ordens por Série

### Séries `store_total` e `FOODS`
As vendas de alimentos (*Foods*) representam a maior fatia do volume total da loja, o que faz com que a série `store_total` herde praticamente a mesma assinatura visual e de autocorrelação. Por isso, a mesma ordem foi ajustada para ambas.

*   **Componente Sazonal $(P, D, Q)_7$**: Os picos nos *lags* sazonais (múltiplos de 7) no ACF têm um decaimento muito lento, indicando não-estacionariedade sazonal. Por isso, aplicamos uma diferença sazonal ($D=1$). No PACF, há um pico claro e isolado no *lag* 7, justificando um termo autorregressivo sazonal ($P=1$). Para suavizar o erro da diferenciação, incluímos um termo de média móvel sazonal ($Q=1$). Logo, a ordem sazonal escolhida é **$(1, 1, 1)_7$**.
*   **Componente Não-Sazonal $(p, d, q)$**: Ignorando os picos de 7 em 7, a base do gráfico ACF também decai lentamente nas primeiras defasagens, o que acusa uma não-estacionariedade na média geral. Corrigimos isso com uma diferença de primeira ordem ($d=1$). O PACF regular corta abruptamente após os primeiros *lags*, sugerindo $p=1$. Adicionamos $q=1$ para lidar com a dependência de curto prazo remanescente.
*   **Ordem Final**: $\text{SARIMA}(1, 1, 1)(1, 1, 1)_7$

### Série `HOBBIES`
Esta categoria possui um comportamento estrutural mais ruidoso e uma quebra de patamar visível entre 2012 e 2013. O debate do grupo girou em torno de aplicar ou não a diferenciação não-sazonal ($d$).

*   **A Dúvida entre $d=0$ e $d=1$**: Diferente de *Foods*, o ACF de *Hobbies* nos *lags* intra-semanais (1 a 6) cai para perto de zero de forma mais rápida. Isso nos levou a considerar um modelo $\text{SARIMA}(1, 0, 1)(1, 1, 1)_7$, assumindo que a série já fosse quase estacionária dentro das semanas. No entanto, devido à quebra de patamar histórico (aumento do nível de vendas no meio da série), a ausência de uma diferenciação ($d=0$) faria o modelo prever o futuro retornando para uma média histórica irreal. 
*   **Decisão**: Para blindar o modelo contra quebras estruturais e manter a consistência metodológica com as demais séries, optamos por forçar a primeira diferença ($d=1$). O componente sazonal manteve-se idêntico devido à presença confirmada dos picos semanais no ACF/PACF.
*   **Ordem Final**: $\text{SARIMA}(1, 1, 1)(1, 1, 1)_7$

---

**Nota Técnica:** Como é comum na biblioteca `statsmodels` ao lidar com muitos anos de dados diários, os parâmetros `enforce_stationarity=False` e `enforce_invertibility=False` foram utilizados durante o `.fit()` para auxiliar a convergência do algoritmo de otimização matemática, garantindo que o modelo ajustasse a tempo e evitasse erros de *LinAlgError*.