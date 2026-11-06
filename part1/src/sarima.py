import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.data._config import BASE_PATH
from src.data.reader import DataReader, SERIES

# Dicionário com as ordens que você definiu na análise visual
# Formato: "nome_da_serie": ((p, d, q), (P, D, Q, m))
SARIMA_ORDERS = {
    "store_total": ((1, 1, 1), (1, 1, 1, 7)),
    "FOODS":       ((1, 1, 1), (1, 1, 1, 7)),
    "HOBBIES":     ((1, 0, 1), (1, 1, 1, 7)), # ou (1, 1, 1) dependendo dos testes
}

OUTPUT_PATH = BASE_PATH / "outputs"

def ajustar_e_prever_sarima(train: DataReader, datas_validacao: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Ajusta modelos SARIMA apenas no conjunto de treino e prevê o horizonte de validação.
    Retorna o dataframe formatado com colunas: date, series, modelo, yhat.
    """
    # Garantir que estamos prevendo imediatamente após o treino
    if datas_validacao[0] != train.get_dates()[-1] + pd.Timedelta(days=1):
        raise ValueError("O horizonte de validação deve começar no dia seguinte ao fim do treino.")

    horizonte = len(datas_validacao)
    previsoes = []

    for nome in SERIES:
        print(f"Treinando SARIMA para {nome}...")
        y_treino = train.get_series(nome)
        
        ordem, ordem_sazonal = SARIMA_ORDERS[nome]
        
        # Ajustar o modelo APENAS no treino
        # enforce_stationarity e enforce_invertibility falsos ajudam a evitar erros de convergência
        modelo = SARIMAX(
            y_treino, 
            order=ordem, 
            seasonal_order=ordem_sazonal,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        resultado = modelo.fit(disp=False)
        
        # Prever o horizonte desejado
        yhat = resultado.forecast(steps=horizonte)
        
        # Guardar resultados no formato esperado
        df_prev = pd.DataFrame({
            "date": datas_validacao,
            "series": nome,
            "modelo": "sarima", # Conforme exigido pelo enunciado: "sarima" ou "arima"
            "yhat": yhat.values
        })
        
        previsoes.append(df_prev)
        print(f"Previsão concluída para {nome}.")

    # Juntar todas as séries
    return pd.concat(previsoes, ignore_index=True)

if __name__ == "__main__":
    # Carregar dados
    train = DataReader("train")
    valid = DataReader("validation")
    
    # Criar pasta se não existir
    OUTPUT_PATH.mkdir(exist_ok=True)
    
    # Gerar previsões
    print("Iniciando modelagem SARIMA...")
    df_sarima = ajustar_e_prever_sarima(train, valid.get_dates())
    
    # Salvar em CSV
    caminho_saida = OUTPUT_PATH / "previsoes_sarima.csv"
    df_sarima.to_csv(caminho_saida, index=False)
    print(f"Previsões SARIMA salvas com sucesso em: {caminho_saida}")