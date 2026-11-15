from ...data.load_data import load_data
from ...data.prepare_data import (
    compute_train_stats,
    build_exog_features,
    build_train_base_recursive,
    build_validation_base_recursive
)
from ..models_utils import predict_model, compute_metrics, save_outputs
from ...config import (
    SERIE, VAL_DATES, OUTPUT_DIR
)
from .recursive_mlp_class import RecursiveMLPForecaster


if __name__ == "__main__":    
    # CV selected HP
    
    hyperparameters = {
        'hidden_layer_sizes': (32,),
        'alpha': 0.0001,
        'learning_rate_init': 0.001,
        'early_stopping': False
    }

    train_df, val_df, cov_df = load_data()

    stats = compute_train_stats(train_df, cov_df)
    exog_feats = build_exog_features(cov_df, stats)

    # Assume build_train_base uses build_recursive_features internally, 
    # dropping NaNs caused by the 28-day shift.
    X_train_df, y_train_std = build_train_base_recursive(train_df, exog_feats, stats)

    # Build validation features (exogenous only for now, lags will be populated dynamically)
    X_validation = build_validation_base_recursive(exog_feats, y_train_std, stats, VAL_DATES)

    model = RecursiveMLPForecaster(params=hyperparameters)
    model.fit(X_train_df, y_train_std)

    # (a) Via predict_model do pipeline (se ele despadroniza):
    y_hat_final = predict_model(X_validation, stats, model, X_validation.columns, True)

    y_true = val_df.set_index("date").loc[VAL_DATES, SERIE].to_numpy()
    validation_metrics = compute_metrics(
        y_true, 
        y_hat_final.to_numpy(), 
        train_df[SERIE].to_numpy()
    )

    OUTPUT_DIR = OUTPUT_DIR.parent.parent / 'part4' / 'outputs'
    save_outputs(y_hat_final, VAL_DATES, validation_metrics, model_name='rede', output_dir=OUTPUT_DIR)
    print(validation_metrics)