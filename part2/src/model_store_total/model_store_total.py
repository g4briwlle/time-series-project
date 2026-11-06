from sklearn.linear_model import LinearRegression
import pandas as pd

import datetime as dt
from typing import Dict, List

from .load_data_st import carregar_dados
from .prepare_data_st import (
    calcular_estatisticas_treino,
    build_exog_features,
    construir_base_treino,
    construir_base_validacao,
    feature_columns
)
from .config_store_total import (
    VAL_END,
    VAL_START,
    SERIE,
    BASE_PATH
)
from ..model_direto_hobbies import calcular_metricas

def save_outputs(y_hat: pd.Series, val_dates: pd.DatetimeIndex, metrics: Dict[str, float]):
    OUTPUT_PATH = BASE_PATH / 'outputs'
    predictions_path = OUTPUT_PATH / 'previsoes_linear_store_total.csv'
    metrics_path = OUTPUT_PATH / 'metricas_linear_store_total.csv'
    
    pd.DataFrame({
        "date": val_dates.strftime("%Y-%m-%d"), "series": SERIE, "modelo": "linear", "yhat": y_hat
    }).to_csv(predictions_path, index=False)
    
    pd.DataFrame([{
        "series": SERIE, "modelo": "linear", "mae": metrics['mae'], "rmse": metrics['rmse'], "mase": metrics['mase']
    }]).to_csv(metrics_path, index=False)

if __name__ == "__main__":
    train_df, val_df, cov_df = carregar_dados()
    
    # 1) Estatísticas do treino (antes de construir features)
    stats = calcular_estatisticas_treino(train_df, cov_df)

    # 2) Features exógenas sobre treino + validação (uma única vez)
    exog_feats = build_exog_features(cov_df, stats)

    # 3) Bases de treino e validação
    X_train_df, y_train_std = construir_base_treino(train_df, exog_feats, stats)
    X_validation = construir_base_validacao(exog_feats, y_train_std)

    # 4) Sanidade: mesmas colunas, mesma ordem
    cols = feature_columns(X_train_df)
    assert cols == list(X_validation.columns), \
        f"Schema divergente: train={cols}, val={list(X_validation.columns)}"

    # 5) OLS
    model = LinearRegression()
    model.fit(X_train_df[cols].to_numpy(), y_train_std.to_numpy())

    # 6) Previsão na validação (despadroniza)
    z_hat = model.predict(X_validation[cols].to_numpy())
    y_hat = pd.Series(
        z_hat * stats["y_std"] + stats["y_mean"],
        index=X_validation.index,
        name="yhat",
    )

    val_dates = pd.date_range(VAL_START, VAL_END, freq="D")
    y_true = val_df.set_index("date").loc[val_dates, SERIE].to_numpy()

    validation_metrics = calcular_metricas(y_true, y_hat.to_numpy(), train_df[SERIE].to_numpy())
    
    print(validation_metrics)
    
    save_outputs(y_hat, val_dates, validation_metrics)