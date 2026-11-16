# Conclusão e Análise de Resultados - Task 2 (Séries Temporais)



## 1. Estratégia de Modelagem e Features

Os hiperparâmetros foram escolhido unicamente pelo erro de validação cruzada expansiva (*expanding-window CV* com 4 *folds*) em janelas que terminam impreterivelmente até a origem oficial (`2016-03-27`), garantindo **zero vazamento de dados (no data leakage)**.

**Features (Covariáveis):**
Para os modelos de rede, incorporei as covariáveis externas presentes em `covariaveis.csv` que já eram conhecidas na data de origem:
*   Variáveis de calendário (dia da semana, mês).
*   Eventos e datas comemorativas (`event_name`, `event_type`).
*   Indicador de benefício `SNAP_CA`.
*   Preços padronizados (com médias e desvios retirados estritamente da base de treino).

Na **estratégia recursiva**, somou-se a essas covariáveis a própria dinâmica endógena da série:  *lags* específicos (1 a 7, 14, 21, 28) e janelas móveis de estatísticas (*rolling mean* e *rolling std* de 7 e 28 dias). No horizonte de predição, esses *lags* foram iterativamente atualizados com a previsão do próprio modelo $\hat{y}$ (1-step-ahead recursivo).

## 2. A Escolha da Arquitetura: LSTM vs. MLP

Para a classe de "Redes Neurais", optei por uma **LSTM** (*Long Short-Term Memory*), aplicada através do PyTorch. Essa escolha foi motivada pela **Natureza Sequencial dos Dados** e o **Tamanho do Dataset (~1900 pontos)**.
1. **Por que não uma MLP? (Natureza Sequencial):** Uma MLP processa *lags* temporais e *features* exógenas como um vetor tabular estático (sem ordem natural explícita). A LSTM processa janelas causais (*seq_len*) iterativamente, mantendo um estado oculto (memória) que consegue extrair dependências temporais de forma muito mais rica.

2. **Por que não uma RNN Vanilla? (Memória de Longo Prazo):** Redes Recorrentes simples (RNNs) sofrem drasticamente com o problema do desvanecimento do gradiente (*vanishing gradient*). Elas têm grande dificuldade em reter informações de passos temporais mais distantes (como a sazonalidade mensal ou janelas de 28 dias). O mecanismo de *gates* (portões de esquecimento e atualização) da LSTM foi criado especificamente para resolver esse problema.

3. **Por que não um Transformer? (Data-Hungry e Overfitting):** Embora os Transformers sejam o atual estado da arte para sequências, eles são arquiteturas *data-hungry* (exigem volumes massivos de dados para calibrar os pesos do *Self-Attention*) e não possuem viés indutivo natural para a ordem do tempo (dependem de codificação posicional). Para uma série relativamente curta de \~1900 pontos diários, um Transformer correria um risco altíssimo de superajustamento (*overfitting*) severo, memorizando o ruído do treino. A LSTM, sendo mais compacta, atua como um regularizador natural ideal para esse volume intermediário de dados.

Na série **HOBBIES**, optei pela **Estratégia Recursiva** para a LSTM, pois obteve um `MASE` validado por CV de `~0.73` no treino, derrotando a estratégia Direta.

## 3. Comparação com Baselines (Naive Sazonal e SARIMA)


A avaliação final foi pautada pelas métricas computadas de forma cega no horizonte de validação de 28 dias (`2016-03-28` a `2016-04-24`), com a métrica primária de ranqueamento sendo o MASE (escalado pelo MAE *in-sample* do Naive Sazonal do período 7), no arquivo final de todas as métricas.


Em síntese, a integração de covariáveis com abordagens que tratam a temporalidade de maneira explícita (como a LSTM) justificaram o aumento de custo computacional em relação aos modelos da Task 1.