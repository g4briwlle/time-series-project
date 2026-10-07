"""
Final script of project that should produce the final metrics.csv as a requirement.
"""
import pandas as pd

from typing import Callable

from src.data.reader import DataReader

EXPECTED_COLUMNS = [
    'series',
    'modelo',
    'mae',
    'rmse',
    'mase',
]
EXPECTED_BASELINES = [
    'naive',
    'naive_sazonal',
    'media',
    'drift',
]

def mock_function(validation_data: pd.DataFrame) -> pd.DataFrame:
    print("Creating fake metrics dataframe for mock")
    
    fake_metrics_df = validation_data.copy()
    
    fake_metrics_df = fake_metrics_df.groupby('series')['y'].mean().reset_index()
    
    fake_metrics_df['mae'] = fake_metrics_df['y']
    fake_metrics_df['rmse'] = fake_metrics_df['y']
    fake_metrics_df['mase'] = fake_metrics_df['y']
    
    fake_metrics_df = fake_metrics_df.drop(columns=['y'])
    
    series = fake_metrics_df['series'].unique().tolist()
    
    multiindex = pd.MultiIndex.from_product([series, EXPECTED_BASELINES], names=['series', 'modelo'])
    
    fake_metrics_df = (
        fake_metrics_df.set_index('series')
        .reindex(multiindex)
        .reset_index()
    )
    
    return fake_metrics_df

def assert_metrics_df(metrics_df: pd.DataFrame):
    assert 'mae' in metrics_df.columns, "Dataframe de métricas não contem a coluna 'mae' necessária"
    assert 'rmse' in metrics_df.columns, "Dataframe de métricas não contem a coluna 'rmse' necessária"
    assert 'mase' in metrics_df.columns, "Dataframe de métricas não contem a coluna 'mase' necessária"
    
    assert set(EXPECTED_COLUMNS) == set(metrics_df.columns.to_list()), "Dataframe de métricas não tem as colunas necessárias ['series', 'modelo', 'mae', 'rmse', 'mase']"

    metrics_df_models = metrics_df['modelo'].unique().tolist()

    for expected_baseline in EXPECTED_BASELINES:
        assert expected_baseline in metrics_df_models, f"Dataframe de métricas não contém baseline '{expected_baseline}'"

def generate_metrics_file(metrics_generating_function: Callable):
    validation_data = DataReader('validation').to_long()
    
    metrics_df = metrics_generating_function(validation_data)
    
    assert_metrics_df(metrics_df)
    
    print(metrics_df)
    
if __name__ == "__main__":
    generate_metrics_file(mock_function)