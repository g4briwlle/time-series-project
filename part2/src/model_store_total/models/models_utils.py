import pandas as pd
import numpy as np

from typing import Dict

from ..config import SERIE, OUTPUT_DIR, VAL_DATES
from ..data.load_data import carregar_dados
from ..data.prepare_data import compute_train_stats, get_all_data, get_all_data_recursive



# --- Functions --------------------------------------------------------
def fit_and_predict(model, X_train_df, y_train_std, X_validation):
    """
    Universal function for fitting and predicting with a model
    """

    # Exclude the date index/column if it's still present
    cols = [c for c in X_train_df.columns if c not in ("date", "index")]
    
    X_train = X_train_df[cols].to_numpy()
    y_train = y_train_std.to_numpy()
    X_val   = X_validation[cols].to_numpy()
    
    model.fit(X_train, y_train)
    return model.predict(X_val)

def compute_metrics(y_true: np.ndarray, yhat: np.ndarray, y_train_raw: np.ndarray) -> Dict[str, float]:
    """Cálculo das métricas padronizado pela Task 2."""
    err = yhat - y_true
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err**2).mean()))
    mase_scale = float(np.abs(y_train_raw[7:] - y_train_raw[:-7]).mean())
    mase = mae / mase_scale if mase_scale > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "mase": mase}

def save_outputs(
    y_hat: pd.Series,
    val_dates: pd.DatetimeIndex,
    metrics: Dict[str, float],
    model_name: str
):
    predictions_path = OUTPUT_DIR / f'previsoes_{model_name}_store_total.csv'
    metrics_path = OUTPUT_DIR / f'metricas_{model_name}_store_total.csv'
    
    pd.DataFrame({
        "date": val_dates.strftime("%Y-%m-%d"), "series": SERIE, "modelo": "linear", "yhat": y_hat
    }).to_csv(predictions_path, index=False)
    
    pd.DataFrame([{
        "series": SERIE, "modelo": "linear", "mae": metrics['mae'], "rmse": metrics['rmse'], "mase": metrics['mase']
    }]).to_csv(metrics_path, index=False)
    
def predict_model(X_val_h, stats, model, cols):
    z_hat = model.predict(X_val_h[cols].to_numpy())
    y_hat = pd.Series(
        z_hat * stats["y_std"] + stats["y_mean"],
        index=X_val_h.index,
        name="yhat",
    )
    
    return y_hat

def tree_and_boosting_pipeline(model, model_name):
    """
    Common model fit, predict, report pipeline for using in different models.
    """
    
    # --- Data steps -------------------------------------------------------
    train_df, _, cov_df = carregar_dados()
    stats = compute_train_stats(train_df, cov_df)
    X_train_h, y_train_h, X_val_h, y_true = get_all_data()
    y_train_naive = train_df[SERIE].to_numpy()

    cols = [c for c in X_train_h.columns if c not in ("date", "index")]

    # --- Fit and Predict --------------------------------------------------
    print(f"--- Training {model_name} ---")
        
    # 5) Fit
    model.fit(X_train_h[cols].to_numpy(), y_train_h.to_numpy())

    # 6) Previsão na validação (despadroniza)
    y_hat = predict_model(X_val_h, stats, model, cols)

    # --- Report -----------------------------------------------------------

    # 7) Métricas
    validation_metrics = compute_metrics(
        y_true, 
        y_hat.to_numpy(), 
        y_train_naive
    )

    # 8) Salvar resultados (assumindo que save_outputs recebe o nome do modelo 
    # ou que você adapta para salvar em tabelas diferentes)
    save_outputs(y_hat, VAL_DATES, validation_metrics, model_name=model_name)
    print(validation_metrics)
    
def tree_and_boosting_recursive_pipeline(model, model_name):
    """
    Trains the model on 1-step ahead data and predicts recursively over the validation horizon.
    """
    
    train_df, _, cov_df = carregar_dados()
    stats = compute_train_stats(train_df, cov_df)
    X_train, y_train, X_val, y_true, stats = get_all_data_recursive()
    y_train_naive = train_df[SERIE].to_numpy()
    
    cols = [c for c in X_train.columns if c not in ("date", "index")]
    
    print(f"--- Training {model_name} (Recursive) ---")
    
    # Fit strictly on training data
    model.fit(X_train[cols].to_numpy(), y_train.to_numpy())

    y_hat_list = []
    X_val_recursive = X_val.copy().reset_index(drop=True)
    
    # Maintain a history array of standardized targets to dynamically compute features
    # This prevents any validation 'y' from leaking into the predictions.
    history = list(y_train.to_numpy())

    for i in range(len(X_val_recursive)):
        # 1. Dynamically update features for step i based on the growing history
        X_val_recursive.loc[i, 'lag_1'] = history[-1]
        X_val_recursive.loc[i, 'lag_2'] = history[-2]
        X_val_recursive.loc[i, 'lag_3'] = history[-3]
        X_val_recursive.loc[i, 'lag_4'] = history[-4]
        X_val_recursive.loc[i, 'lag_5'] = history[-5]
        X_val_recursive.loc[i, 'lag_6'] = history[-6]
        X_val_recursive.loc[i, 'lag_7'] = history[-7]
        X_val_recursive.loc[i, 'lag_14'] = history[-14]
        X_val_recursive.loc[i, 'lag_21'] = history[-21]
        X_val_recursive.loc[i, 'lag_28'] = history[-28]
        
        X_val_recursive.loc[i, 'rolling_mean_7'] = np.mean(history[-7:])
        X_val_recursive.loc[i, 'rolling_mean_28'] = np.mean(history[-28:])
        X_val_recursive.loc[i, 'rolling_std_7'] = np.std(history[-7:], ddof=1)
        
        # 2. Extract current row and predict
        current_features = X_val_recursive.loc[[i], cols]
        current_pred = model.predict(current_features.to_numpy())[0]
        
        # 3. Store prediction and append to history for the next iteration
        y_hat_list.append(current_pred)
        history.append(current_pred)
        
    # Reverse the standardization (assumes predict_model logic is replaced by this step)
    # y_hat_final should apply the mean and std used during the compute_train_stats step
    y_hat_final = (np.array(y_hat_list) * stats['y_std']) + stats['y_mean']
    
    # Metrics
    # Assuming y_train_naive uses the unstandardized original training series
    validation_metrics = compute_metrics(
        y_true, 
        y_hat_final, 
        y_train_naive
    )

    save_outputs(y_hat_final, VAL_DATES, validation_metrics, model_name=model_name)
    print(validation_metrics)