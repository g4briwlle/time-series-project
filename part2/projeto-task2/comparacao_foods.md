# Modelo linear — FOODS

O objetivo foi prever as vendas diárias da categoria FOODS da loja
Walmart CA_1 para os 28 dias de 28/03/2016 a 24/04/2016.

O ajuste utilizou exclusivamente os dados de treinamento, de
29/01/2011 a 27/03/2016, totalizando 1885 observações. As vendas reais
da validação oficial foram utilizadas somente para calcular as
métricas finais.

Foram utilizadas as seguintes variáveis:

- Dia da semana e mês, derivados da coluna `date`.
- `snap_CA`: indicador do calendário SNAP na Califórnia. 
    O crédito SNAP é destinado à compra de alimentos, pois seu calendário pode se relacionar com mudanças nas vendas de alimentos.
- `event_name_1`, `event_type_1`, `event_name_2` e `event_type_2`:
  eventos e seus tipos.
- `preco_FOODS`: preço médio semanal da categoria.

Foram comparadas regressão linear comum, Ridge e Lasso, utilizando
as mesmas variáveis de entrada e os mesmos cortes temporais.

Utilizamos a estratégia direta porque as entradas do modelo são covariáveis 
conhecidas na origem da previsão, sem defasagens das vendas. Como limitação, 
o modelo não incorpora explicitamente a dinâmica das vendas recentes.

A regressão comum serviu como referência sem regularização.
Ridge e Lasso foram incluídos para avaliar se a regularização
melhoraria a previsão diante de variáveis categóricas e informações
relacionadas, como nomes e tipos de eventos.

Para Ridge e Lasso, foram testados os valores de `alpha`:
0,01; 0,1; 1; 10; 100.

A seleção foi realizada pelo menor MAE médio em três cortes
temporais internos, com treinamento expansivo e horizonte de
28 dias por corte.

| Modelo | Alpha | MAE médio interno | Desvio-padrão do MAE |
|---|---:|---:|---:|
| Linear comum | Não se aplica | 254,653 | 42,620 |
| Ridge | 0,1 | 254,549 | 42,779 |
| Lasso | 0,1 | 253,536 | 47,160 |

O modelo selecionado foi Lasso com `alpha=0.1`. Sua vantagem foi
pequena. Esses resultados não demonstram uma superioridade ampla, 
mas determinam a escolha segundo o critério estabelecido.

O desvio-padrão apresentado mede a variação do MAE entre os três
cortes; não representa um intervalo de confiança.

Após a seleção, o Lasso foi ajustado novamente utilizando todo o
treinamento. A validação oficial não foi usada para escolher a
família do modelo nem o valor de `alpha`.


O mesmo modelo prevê cada dia do horizonte a partir do calendário,
dos eventos, do SNAP e do preço correspondente à data. 
Como não foram utilizadas defasagens
das vendas, nenhuma previsão precisou retornar como entrada para
o dia seguinte.

As vendas reais da validação não foram utilizadas como entradas.
Nos arquivos de entrega, o modelo recebe o nome `linear`,
conforme o formato exigido, embora a variante escolhida seja Lasso.

## Avaliação oficial e comparação com a Task 1

As métricas foram calculadas nos mesmos 28 dias da validação oficial.

Para calcular a escala do MASE, cada venda de treinamento foi
comparada com a venda observada sete dias antes. A escala é a
média dos valores absolutos dessas diferenças.

| Modelo | MAE | RMSE | MASE |
|---|---:|---:|---:|
| Naive sazonal | 343,857 | 420,116 | 1,005172 |
| SARIMA | 180,021 | 225,195 | 0,526243 |
| Linear — Lasso, alpha=0,1 | 146.62063 | 177.570673 | 0.428605 |

Na validação oficial, o modelo linear Lasso com alpha=0,1
apresentou os menores erros nas três métricas: MAE de 146,621,
RMSE de 177,571 e MASE de 0,428605.

Em comparação com o naive sazonal, o MAE caiu aproximadamente
57,4%. Em relação ao SARIMA, a redução foi de 18,6%.
O Lasso também superou ambos em RMSE e MASE, não perdendo
para o naive sazonal em nenhuma das métricas avaliadas.

O MAE indica um erro absoluto médio de aproximadamente
147 unidades de alimentos por dia. O MASE de 0,428605
significa que esse erro corresponde a aproximadamente 42,9%
da escala sazonal calculada no treinamento, de 342,087859.

Esses resultados mostram melhor desempenho do Lasso neste
horizonte de 28 dias. Não demonstram que ele será superior
em todos os períodos nem permitem atribuir a melhora a uma
covariável específica.

O MAE da comparação interna não deve ser comparado diretamente 
com as métricas oficiais da Task 1, pois os períodos avaliados 
são diferentes.
