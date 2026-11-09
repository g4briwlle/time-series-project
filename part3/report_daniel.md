# Texto de implementação dos modelos para série `store_total`

## 1. Desenho da validação cruzada temporal
Utilizei walk-forward com janela expansível, com três folds não sobrepostos, todos terminando em ou antes da origem oficial (2016-03-27):

| Fold | Treino (≤) | Validação |
|----|----------|---------|
| 0 | 2016-01-02 | 2016-01-03 → 2016-01-30 |
| 1 | 2016-01-30 | 2016-01-31 → 2016-02-27 |
| 2 | 2016-02-27 | 2016-02-28 → 2016-03-27 |

Cada fold de validação tem exatamente 28 dias — o mesmo comprimento do horizonte oficial — e o último fold é adjacente à origem, o que o torna o mais informativo para ranquear configurações. Em cada fold, a média e o desvio usados para padronizar a série e os preços, as estatísticas das covariáveis e as features de lag são calculados apenas com o treino do fold, espelhando a restrição do enunciado ("a média e o desvio que padronizam a série, e os do preço, saem só do treino"). A validação do fold vem de treino.csv filtrado pela janela correspondente; o validacao.csv oficial nunca entra no loop de busca.

## 2. A estratégia como hiperparâmetro

A escolha entre estratégia direta ou recursiva é um hiperparâmetro do pipeline: ela não altera o espaço de busca dos demais parâmetros, apenas a forma como o modelo é treinado e como as previsões são encadeadas. Incluí-la como mais um eixo da mesma grade permite que um único loop interno compare, sob o mesmo critério, todas as combinações de (estratégia, hiperparâmetros). Um nível adicional só faria sentido se quiséssemos estimar separadamente a incerteza da etapa de seleção — o que o enunciado não pede e que multiplicaria o custo sem ganho prático.

Adotamos, portanto, um único nível de validação cruzada temporal, com o eixo estrategia na grade. Isso satisfaz a exigência do enunciado (escolha justificada, feita sem vazar a validação oficial) e mantém a comparação entre direta e recursiva limpa: ambas percorrem a mesma grade, com o mesmo critério de seleção.

## 3. Grades compartilhadas
As grades são definidas uma única vez por classe de modelo e reutilizadas para as duas estratégias:

```python
BOOSTING_GRID = {
    "estrategia":        ["direto", "recursivo"],
    "max_iter":          [200, 400],
    "learning_rate":     [0.05, 0.1],
    "max_depth":         [3, 4],
    "min_samples_leaf":  [10, 20],
    "l2_regularization": [0.0, 0.5, 1.5],
}

ARVORE_GRID = {
    "estrategia":       ["direto", "recursivo"],
    "max_depth":        [4, 6, 8],
    "min_samples_leaf": [10, 20, 40],
    "ccp_alpha":        [0.0, 1e-4],
}
```
A escolha dos eixos reflete as regularizações relevantes para cada classe:

arvore (DecisionTreeRegressor). max_depth e min_samples_leaf controlam a complexidade estrutural; ccp_alpha é a poda pós-ajuste por cost-complexity pruning, o único shrinkage explícito disponível na árvore. Incluí-lo permite que a estratégia recursiva, mais sujeita a variância, seja penalizada se for o caso.

boosting (HistGradientBoostingRegressor). max_iter e learning_rate governam a taxa de aprendizado e o número de árvores; max_depth e min_samples_leaf controlam a capacidade de cada árvore; l2_regularization é o shrinkage L2 sobre as folhas, e é o eixo central para testar a hipótese de que a recursiva precisa de mais regularização.

early_stopping=False foi fixado em ambos os casos. O split interno do HistGradientBoostingRegressor é aleatório, não temporal; embora todos os dados sejam ≤ 2016-03-27 (portanto sem vazamento real), o split aleatório contraria o espírito temporal e adiciona variância. O tamanho do modelo é controlado por max_iter + learning_rate.

A hipótese motivadora — a recursiva tende a precisar de mais regularização porque acumula lags altamente correlacionados e propaga erros passo a passo — é tratada como conjectura a ser testada, não como priori codificada. Se a CV interna eleger l2_regularization=1.5 e ccp_alpha=1e-4 para a recursiva e valores menores para a direta, isso será reportado como resultado empírico. Se eleger os mesmos valores para ambas, a conjectura é refutada neste conjunto — e isso também é reportado. Em nenhum dos casos o código impõe assimetria entre estratégias, o que preserva a comparabilidade.

## 4. Critério de seleção e fluxo
As três dobras temporais são preparadas uma única vez (prepare_folds_direct / prepare_folds_recursive), com features e estatísticas estritamente restritas ao treino de cada fold.

Toda a grade é avaliada em paralelo (grid_search_direct / grid_search_recursive), instanciando um estimador novo por configuração e por fold.

Para cada configuração, calcula-se o MAE em cada fold e normaliza-se pelo MAE in-sample do naive sazonal semanal calculado no treino daquele fold — reproduzindo a definição de MASE da Task 1.

O critério de ranqueamento é a média do MASE entre os três folds. A configuração com menor média vence dentro de cada estratégia.

As duas melhores configurações — uma por estratégia — são comparadas pelo mesmo critério; a de menor média de MASE é escolhida como configuração final da classe naquela série. A escolha entre direta e recursiva é, portanto, o resultado direto da mesma busca, sem loop adicional.

O vencedor é re-treinado na janela de treino completa (≤ 2016-03-27), com as estatísticas de padronização recalculadas sobre todo o treino, e só então usado para prever a validação oficial.

A estrutura resultante é:

```text
para cada classe (arvore, boosting):
grade compartilhada com eixo "estrategia"
para cada fold temporal (3 folds, todos ≤ 2016-03-27):
    preparar features com estatísticas do treino do fold
avaliar todas as configurações (direta + recursiva) em paralelo
ranquear por MASE médio entre folds
escolher a melhor configuração global (uma única, cruzando estratégias)
re-treinar vencedor no treino completo
prever validação oficial
```

**Modelos finais após cross-validation**

| Classe | Estratégia selecionada | MASE médio (Cross Validation) | Hiperparâmetros selecionados |
|---|---:|---:|---|
| `arvore` | **recursivo** (`r`) | **0.7089659141707365** | `max_depth=4`, `min_samples_leaf=10`, `ccp_alpha=0.0` |
| `boosting` | **direto** (`d`) | **0.6359961173161627** | `max_iter=200`, `learning_rate=0.1`, `max_depth=3`, `min_samples_leaf=20`, `l2_regularization=0.0` |
