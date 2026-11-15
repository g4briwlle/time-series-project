from ..data.load_data import load_data
from ..data.prepare_data import (
    compute_train_stats,
    build_exog_features,
    build_train_base_recursive,
    build_validation_base_recursive,
)
from ...model_store_total.models.models_utils import predict_model, compute_metrics, save_outputs
from ..config import SERIE, VAL_DATES, OUTPUT_DIR
from .recursive_lstm_class import RecursiveLSTMForecaster

hyperparameters = {
    "hidden_size": 32,
    "num_layers": 1,
    "seq_len": 7,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 40,
    "batch_size": 64,
}

if __name__ == "__main__":
    train_df, val_df, cov_df = load_data()

    stats = compute_train_stats(train_df, cov_df)
    exog_feats = build_exog_features(cov_df, stats)

    X_train_df, y_train_std = build_train_base_recursive(train_df, exog_feats, stats)
    X_validation = build_validation_base_recursive(exog_feats, y_train_std, stats, VAL_DATES)

    model = RecursiveLSTMForecaster(params=hyperparameters)
    model.fit(X_train_df, y_train_std)

    y_hat_final = predict_model(X_validation, stats, model, X_validation.columns, True)
    y_true = val_df.set_index("date").loc[VAL_DATES, SERIE].to_numpy()
    validation_metrics = compute_metrics(
        y_true, y_hat_final.to_numpy(), train_df[SERIE].to_numpy()
    )

    out_dir = OUTPUT_DIR.parent.parent / "part4" / "outputs"
    save_outputs(y_hat_final, VAL_DATES, validation_metrics, model_name="rede", output_dir=out_dir)
    print(f"[{SERIE} - Recursive LSTM] {validation_metrics}")