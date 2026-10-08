# Preparando o ambiente

### Criando e ativando ambiente virtual

Dê preferência por utilizar o Python 3.14 para esse projeto. Crie um ambiente virtual rodando:

```bash
python3.14 -m venv .venv
```

E inicie o ambiente virtual rodando:

**no Windows**:

```cmd
.\.venv\Scripts\activate.bat
```

**no Linux**:

```cmd
source .venv\bin\activate
```

### Instalando dependências do projeto

Para facilidade, vamos usar apenas o pip para instalar as dependências listadas em requirements. Rode (com o ambiente já ativado):

```bash
pip install -r requirements.txt
```

## Rodando arquivo `run.py`

Para rodar o arquivo `run.py` e gerar tanto `metricas.csv` quanto `previsoes_validacao.csv` execute (com o ambiente virtual criado e ativado, preferencialmente):

```bash
pip install uv && uv pip install -r requirements.txt && python run.py
```

Obs.: Utiliza-se uv para instalação drasticamente mais rápida das dependências.

## Resultados

A comparação dos modelos está em [conclusao.md](conclusao.md).
