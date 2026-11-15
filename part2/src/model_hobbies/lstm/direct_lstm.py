from ..data.load_data import load_data
from ..data.prepare_data import (
    compute_train_stats,
    build_exog_features,
    build_train_base_direct,
    build_validation_base_direct,
    expand_train_horizons,
    add_horizon_to_validation,
)
from .models_utils import predict_model, compute_metrics, save_outputs
from ..config import SERIE, VAL_DATES, OUTPUT_DIR
from .direct_lstm_class import DirectLSTMForecaster

hyperparameters = {
    "hidden_size": 32,
    "num_layers": 1,
    "seq_len": 7,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 35,
    "batch_size": 64,
    "mode": "pooled",
}

if __name__ == "__main__":
    train_df, val_df, cov_df = load_data()

    stats = compute_train_stats(train_df, cov_df)
    exog_feats = build_exog_features(cov_df, stats)

    X_train_df, y_train_std = build_train_base_direct(train_df, exog_feats, stats)
    X_validation = build_validation_base_direct(exog_feats, y_train_std, stats, VAL_DATES)

    X_train_expanded_df, y_train_expanded = expand_train_horizons(X_train_df, y_train_std)
    X_validation_horizon = add_horizon_to_validation(X_validation)

    model = DirectLSTMForecaster(params=hyperparameters, horizons=28, n_jobs=-1)
    model.fit(X_train_expanded_df, y_train_expanded)

    y_hat = predict_model(X_validation_horizon, stats, model, X_validation_horizon.columns, True)
    y_true = val_df.set_index("date").loc[VAL_DATES, SERIE].to_numpy()
    validation_metrics = compute_metrics(y_true, y_hat.to_numpy(), train_df[SERIE].to_numpy())

    print(f"[{SERIE} - Direct LSTM] {validation_metrics}")