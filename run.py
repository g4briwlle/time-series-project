"""
Final script of project that should produce the final metrics.csv as a requirement.
"""
import pandas as pd
import numpy as np

from typing import Callable
from pathlib import Path

from src.data.reader import DataReader

BASE_PATH = Path(__file__).resolve().parent

SERIES = ("store_total", "FOODS", "HOBBIES")
EXPECTED_BASELINES = [
    'naive',
    'naive_sazonal',
    'media',
    'drift',
]

def mock_metrics_generating_function(validation_data: pd.DataFrame) -> pd.DataFrame:
    print("Creating fake metrics dataframe for mock")
    
    fake_metrics_df = validation_data.copy()
    
    fake_metrics_df = fake_metrics_df.groupby('series')['y'].mean().reset_index()
    
    fake_metrics_df['mae'] = fake_metrics_df['y']
    fake_metrics_df['rmse'] = fake_metrics_df['y']
    fake_metrics_df['mase'] = fake_metrics_df['y']
    
    fake_metrics_df = fake_metrics_df.drop(columns=['y'])
    
    series = fake_metrics_df['series'].unique().tolist()
    
    baselines_and_arima = EXPECTED_BASELINES
    baselines_and_arima.append('arima')
    multiindex = pd.MultiIndex.from_product([series, baselines_and_arima], names=['series', 'modelo'])
    
    fake_metrics_df = (
        fake_metrics_df.set_index('series')
        .reindex(multiindex)
        .reset_index()
    )
    
    for column in ['mae', 'rmse', 'mase']:
        fake_metrics_df[column] = np.random.random(len(fake_metrics_df))
    
    return fake_metrics_df

def mock_predictions_generating_function(validation_df: pd.DataFrame) -> pd.DataFrame:
    fake_predictions_df = validation_df.copy()
    
    multiindex_models = EXPECTED_BASELINES
    multiindex_models.append('arima')
    
    series = fake_predictions_df['series'].unique().tolist()
    dates = fake_predictions_df['date'].unique().tolist()
    
    multiindex = pd.MultiIndex.from_product([series, dates, multiindex_models], names=['series', 'date', 'modelo'])
    
    fake_predictions_df = (
        fake_predictions_df.set_index(['series', 'date'])
        .reindex(multiindex)
        .reset_index()
    )
    
    fake_predictions_df = fake_predictions_df.rename(columns={'y': 'yhat'})
    
    return fake_predictions_df

#  --- metricas.csv -----------------------------------------------------
METRICS_PATH = BASE_PATH / 'metricas.csv'
EXPECTED_METRICS_COLUMNS = [
    'series',
    'modelo',
    'mae',
    'rmse',
    'mase',
]

def assert_metrics_df(metrics_df: pd.DataFrame):
    assert 'mae' in metrics_df.columns, "Dataframe de métricas não contem a coluna 'mae' necessária"
    assert 'rmse' in metrics_df.columns, "Dataframe de métricas não contem a coluna 'rmse' necessária"
    assert 'mase' in metrics_df.columns, "Dataframe de métricas não contem a coluna 'mase' necessária"
    assert 'series' in metrics_df.columns, "Dataframe de métricas não contem a coluna 'series' necessária"
    assert 'modelo' in metrics_df.columns, "Dataframe de métricas não contem a coluna 'modelo' necessária"
    
    metrics_df_models = metrics_df['modelo'].unique().tolist()

    for expected_baseline in EXPECTED_BASELINES:
        assert expected_baseline in metrics_df_models, f"Dataframe de métricas não contém baseline '{expected_baseline}'"
    assert (
        ('arima' in metrics_df_models) or
        ('sarima' in metrics_df_models)
    ), "Dataframe de métricas não contém modelo 'arima' ou 'sarima'"
    

def generate_metrics_file(metrics_generating_function: Callable):
    validation_data = DataReader('validation').to_long()
    
    metrics_df = metrics_generating_function(validation_data)
    
    assert_metrics_df(metrics_df)
    
    print(metrics_df)
    
    metrics_df.to_csv(METRICS_PATH, index=False)


# --- previsoes_validacao.csv ------------------------------------------
PREDICTIONS_PATH = BASE_PATH / 'previsoes_validacao.csv'
EXPECTED_VAL_PRED_COLUMNS = (
    'date',
    'series',
    'modelo',
    'yhat',
)

def assert_val_pred_df(val_pred_df: pd.DataFrame):
    # --- Columns ---
    val_pred_columns = val_pred_df.columns.tolist()
    
    for expected_column in EXPECTED_VAL_PRED_COLUMNS:
        assert expected_column in val_pred_columns, f"Dataframe de previsões não possui a coluna '{expected_column}' necessária."
        
    # --- series ---
    val_pred_series = val_pred_df['series'].unique().tolist()
    
    for expected_series in SERIES:
        assert expected_series in val_pred_series, f"Dataframe não possui a série '{expected_series}' necessária."
    
    # --- predictions count ---
    validation_dates = DataReader('validation').get_dates()
    validation_df_size = len(validation_dates)
    
    val_pred_num_dates = len(val_pred_df['date'].unique().tolist())
    
    assert val_pred_num_dates == validation_df_size, "Dataframe de previsões não tem o mesmo tamanho dos dados de validação."
    
    # --- models and baselines ---
    models = val_pred_df['modelo'].unique().tolist()
    
    for expected_baseline in EXPECTED_BASELINES:
        assert expected_baseline in models, f"Dataframe de métricas não contém baseline '{expected_baseline}'"
    
    assert (
        ('arima' in models) or
        ('sarima' in models)
    ), "Não foi detectado modelo SARIMA ou ARIMA em metrics.csv"
    

def generate_validation_predictions_file(
    validation_predictions_generating_function: Callable[[pd.DataFrame], pd.DataFrame]
):
    validation_data = DataReader('validation').to_long()
    
    val_pred_df = validation_predictions_generating_function(validation_data)
    
    assert_val_pred_df(val_pred_df)
    
    val_pred_df.to_csv(PREDICTIONS_PATH, index=False)

if __name__ == "__main__":
    generate_metrics_file(mock_metrics_generating_function)
    generate_validation_predictions_file(mock_predictions_generating_function)