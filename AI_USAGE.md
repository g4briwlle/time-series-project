## Ferramentas utilizadas

- Claude Code (Anthropic): baselines e revisão do PR de diagnóstico.

## Onde ajudou

- **Baselines:** rascunho do código dos quatro baselines (`src/baselines.py`) e do erro in-sample do naive sazonal, que serve de escala do MASE; checagem das previsões contra as fórmulas da aula e contra o `check_task1.py`.
- **Revisão do PR de diagnóstico:** conferência das figuras, do split e do `AI_USAGE.md` contra o `check_task1.py` antes do merge.

## Exemplos de prompts

- **Baselines:** "Draft the code for the mean baseline."
- **Revisão do PR de diagnóstico:** "Help me review the open PR from my colleague. Is there anything wrong? I couldn't see any issue in my own analysis."

## Erros corrigidos

- **Baselines:** a IA fez a função de previsão receber o `DataReader` da validação inteiro, com o y, quando ela só precisava das datas. A revisão do PR pegou isso, e a função passou a receber só `valid.get_dates()`; assim, o código dos baselines não tem acesso ao y da validação. A mesma revisão apontou um comentário fora de lugar no topo do arquivo, que removemos.

## Responsabilidade

A equipe se responsabiliza pelo conteúdo entregue neste trabalho.

A IA ajudou a escrever código, mas o papel humano foi essencial: definimos o escopo de cada tarefa, orientamos a ferramenta ao longo do trabalho e só levamos código para o repositório depois de lê-lo, rodá-lo e conferir os resultados.
