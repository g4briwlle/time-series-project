import pandas as pd
import numpy as np

from typing import Tuple

from ..config import (
    SERIE, VAL_END, VAL_START
)
from .load_data import carregar_dados
from .train_data import *
from .validation_data import *
from .data_utils import *


def expand_train_horizons(
    X_train_df: pd.DataFrame,
    y_train_std: pd.Series,
    max_h: int = 28
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Takes the 1-step train dataframe and expands it to 28 steps.
    Keeps the target and calendar features anchored to the TARGET date,
    and shifts the historical lags forward by h-1 to simulate the origin constraint.
    """
    expanded_X = []
    expanded_y = []
    
    for h in range(1, max_h + 1):
        X_h = X_train_df.copy()
        X_h["h"] = h
        
        # Shift the lag features forward by (h - 1). 
        # If we are predicting date D at horizon h, the forecast was made at D - h + 1.
        # The lag available at that origin was the lag from D - h + 1 - 28.
        lag_cols = [c for c in X_h.columns if "lag" in c.lower()]
        for col in lag_cols:
            X_h[col] = X_h[col].shift(h - 1)
        
        # Target stays completely aligned with the calendar features (DOW, events)
        y_h = y_train_std.copy()
        
        # Drop rows where shifting pushed the lag out of bounds
        valid_mask = X_h[lag_cols[0]].notna()
        
        expanded_X.append(X_h.loc[valid_mask])
        expanded_y.append(y_h.loc[valid_mask])
        
    X_train_h = pd.concat(expanded_X).reset_index(drop=True)
    y_train_h = pd.concat(expanded_y).reset_index(drop=True)
    
    return X_train_h, y_train_h


def add_horizon_to_validation(X_validation):
    """
    Adds h=1..28 to the validation dataframe. 
    Assumes X_validation has exactly 28 rows in chronological order.
    """
    X_val_h = X_validation.copy()
    
    # FIX 2: Freeze sales lags at the origin (first row = 2016-03-27)
    # Identify your lag columns. If they are all like 'y_lag_1', 'y_lag_28', etc.
    lag_cols = [c for c in X_val_h.columns if "lag" in c.lower() or "roll" in c.lower()]
    
    if lag_cols:
        origin_lags = X_val_h.iloc[0][lag_cols].copy()
        X_val_h[lag_cols] = origin_lags.values  # Broadcast to all 28 rows
        
    # Now assign the true horizons 1 to 28
    X_val_h["h"] = np.arange(1, 29)
    return X_val_h

def get_all_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray]:
    """
    Runs the whole data pipeline, returning the final expanded train dataframe
    and series and the validation dataframe with the horizon column.
    
    Returns:
        pd.DataFrame: X_train_expanded_df
        pd.Series: y_train_expanded
        pd.DataFrame: X_validation_horizon
        np.ndarray: y_true
    """
    train_df, val_df, cov_df = carregar_dados()
            
    # 1) Estatísticas do treino (antes de construir features)
    stats = compute_train_stats(train_df, cov_df)

    # 2) Features exógenas sobre treino + validação (uma única vez)
    exog_feats = build_exog_features(cov_df, stats)

    # 3) Bases de treino e validação
    X_train_df, y_train_std = build_train_base_direct(train_df, exog_feats, stats)
    X_validation = build_validation_base_direct(exog_feats, y_train_std, stats)

    # 4) Bases expandidas para os 28 h's
    X_train_expanded_df, y_train_expanded = expand_train_horizons(X_train_df, y_train_std)
    X_validation_horizon = add_horizon_to_validation(X_validation)
    
    val_dates = pd.date_range(VAL_START, VAL_END, freq="D")
    y_true = val_df.set_index("date").loc[val_dates, SERIE].to_numpy()
    
    return (
        X_train_expanded_df,
        y_train_expanded,
        X_validation_horizon,
        y_true,
    )
    
def get_all_data_recursive() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray, dict]:
    """
    Returns data formatted for a 1-step ahead recursive forecasting strategy.
    """
    train_df, val_df, cov_df = carregar_dados()
            
    stats = compute_train_stats(train_df, cov_df)
    exog_feats = build_exog_features(cov_df, stats)

    # Assume build_train_base uses build_recursive_features internally, 
    # dropping NaNs caused by the 28-day shift.
    X_train_df, y_train_std = build_train_base_recursive(train_df, exog_feats, stats)
    
    # Build validation features (exogenous only for now, lags will be populated dynamically)
    X_validation = build_validation_base_recursive(exog_feats, stats)
    
    val_dates = pd.date_range(VAL_START, VAL_END, freq="D")
    y_true = val_df.set_index("date").loc[val_dates, SERIE].to_numpy()
    
    return X_train_df, y_train_std, X_validation, y_true, stats

if __name__ == "__main__":
    train_df, val_df, cov_df = carregar_dados()
        
    # 1) Estatísticas do treino (antes de construir features)
    stats = compute_train_stats(train_df, cov_df)

    # 2) Features exógenas sobre treino + validação (uma única vez)
    exog_feats = build_exog_features(cov_df, stats)

    # 3) Bases de treino e validação
    X_train_df, y_train_std = build_train_base_direct(train_df, exog_feats, stats)
    X_validation = build_validation_base_recursive(exog_feats, stats)
    
    # 4) Bases expandidas para os 28 h's
    X_train_expanded_df, y_train_expanded = expand_train_horizons(X_train_df, y_train_std)
    X_validation_horizon = add_horizon_to_validation(X_validation)
    
    
    print("Train data")
    print(X_train_df.head())
    print(f"Num of rows: {len(X_train_df)}")
    print("Expanded train data")
    print(X_train_expanded_df.head())
    print(f"Num of rows: {len(X_train_expanded_df)}")
    print(50 * "-")
    
    print("Validation data")
    print(X_validation.head())
    print("Validation data with horizon")
    print(X_validation_horizon.head())