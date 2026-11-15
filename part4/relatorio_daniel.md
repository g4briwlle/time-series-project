## Modelos

Os modelos escolhidos para implementação e teste de previsões sobre a série **store_total** são o MLP e o LSTM, o motivo de escolha de cada um explicado em sua respectiva seção.

Para cada modelo foi escolhida uma grid de hiperparâmetros com 3 a 4 hiperparâmetros diferentes, também justificados abaixo.

O treinamento e teste de ambos os modelos compara o desempenho da abordagem direta com a recursiva no conjunto de treinamento, escolhendo a que apresenta menor MASE no conjunto de treinamento. Cada um dos modelos segue uma implementação ligeiramente diferente para a abordagem direta.

### MLP

**Motivo da escolha**

Entre os 4 modelos disponíveis para essa fase, *RNN*, *LSTM* e *Transformer* seguem todos uma lógica parecida de modelar diretamente dependências temporais que dão pesos diferentes a dados ao longo do tempo. Dessas três, a escolha pela LSTM é justificada mais abaixo. Como a *MLP* é fundalmentalmente diferente, por não modelar dependências temporais mas fazer apenas uma tarefa de regressão, me parece um contraponto natural à arquitetura da LSTM a ser testado também na previsão da série **store_total**.

**Abordagem direta**

Serão treinados 28 modelos, cada um especializado na previsão de um horizonte de tempo diferente a partir dos dados de entrada. Essa escolha se dá devido a escassez de dados em relação a quantidade de janelas de previsão e pela relativa facilidade de treinar várias MLPs em paralelo ao invés de tentar ajustar uma única para todos os tamanhos de horizontes diferentes.

**Hiperparâmetros**

| Hiperparâmetro | O que controla | Faixa típica / observação |
|---|---|---|
| **`hidden_layer_sizes`** | Arquitetura (nº de camadas e neurônios) | Comece com uma camada oculta pequena, ex.: `(32,)` ou `(64,)`. Só aumente se houver underfitting evidente no treino. |
| **`alpha`** | Regularização L2 | Controla o overfitting. Valores comuns: `1e-4` a `1e-1` (escala log). |
| **`learning_rate_init`** | Taxa de aprendizado inicial (para `adam`/`sgd`) | `1e-3` é o padrão; testar `1e-4` e `1e-2` ao redor. |

### LSTM

**Motivo da escolha**

Foi observado empiricamente que o modelo *Transformer vanilla* (sem um fine-tuning específico para a tarefa), começa a apresentar ganhos consistentes de desempenho sober a LSTM sem overfitting com cerca de 10.000 pontos de dados, uma quantidade muito maior do que a disponível. A LSTM, em contrapartida, apresenta uma melhor eficiência amostral, com bom desempenho 500 a 1.000 observações [1].

A RNN, por sua vez, também apresenta bastante eficiência amostral, mas, devido ao *vanishing gradient*, não consegue modelar bem uma dependência temporal com mais de 5 passos atrás [2, 3]. Isso leva a um aproveitamente subótimo tanto da sazonalidade semanal conhecida da série de dados quanto da influência de variáveis exógenas, em grande maioria anuais ou mensais.

Com isso, a escolha natural que se contrapõe à MLP para o presente problema de previsão é a **LSTM**, que 

**Abordagem Direta**

A literatura comparativa [4, 5] indica que o MIMO é, na maioria dos cenários sem outliers, tão preciso ou mais preciso que a estratégia direta com modelos separados, além de ser computacionalmente mais eficiente por compartilhar representações entre todos os horizontes. A estratégia direta permanece como alternativa robusta apenas quando os dados contêm outliers significativos, o que não é o caso predominante nesta série.

**Hiperparâmetros**

| Hiperparâmetro | Valores típicos para busca | Observação |
|---|---|---|
| **Número de camadas LSTM** | `[1, 2]` (raramente 3) | Redes mais profundas aumentam o risco de overfitting com ~1.900 pontos. Comece com 1 camada. |
| **Número de neurônios por camada** | `[32, 64, 128]` | 32–128 é a faixa mais comum. Valores acima de 256 são arriscados para o seu tamanho de dados. |
| **Dropout** | `[0.0, 0.1, 0.2, 0.3]` | Essencial para regularizar. Taxas acima de 0.3 podem prejudicar a aprendizagem de dependências longas. |
| **Learning rate (Adam)** | `[0.0005, 0.001, 0.005]` | Valores baixos são mais estáveis; valores altos convergem mais rápido, mas com risco de oscilação. |
| **Tamanho da janela de entrada (*window size*)** | `[7, 14, 21, 28]` | A janela deve capturar pelo menos um ciclo semanal (7 dias). Testar múltiplos de 7 é uma escolha natural para a sazonalidade semanal. |

**Implementação da Estratégia direta** 

Treinar um único modelo MIMO (Multiple Input Multiple Output) que se mostra tão bom quanto ou melhor que 28 modelos, com uma eficiência computacional muito melhor dado o escopo do trabalho.

## Métricas

### Hiperparâmetros escolhidos após grid-search

## Referências
1. https://zenodo.org/records/21288702
2. https://ieeexplore.ieee.org/document/7179007
3. https://theses.hal.science/tel-01470466v2
4. https://link.springer.com/article/10.1007/s10994-025-06875-1#2
5. https://www.socolar.com/Article/Index?aid=100083871144&jid=100000004881