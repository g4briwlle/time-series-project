import pandas as pd

from ...config import (
    DATA_DIR, SERIE, VAL_DATES
)
from .direct_mlp_class import DirectMLPForecaster

# data

train = pd.read_csv(DATA_DIR / "treino.csv", parse_dates=["date"]).set_index("date")
cov = pd.read_csv(DATA_DIR / "covariaveis.csv", parse_dates=["date"]).set_index("date")

hyperparameters = {
    "hidden_layer_sizes": (32,),
    "alpha": 1e-3,
    "learning_rate_init": 1e-3,
}
    
if __name__ == "__main__":
    from ...data.load_data import load_data
    from ...data.prepare_data import (
        compute_train_stats,
        build_exog_features,
        build_train_base_direct,
        build_validation_base_direct,
        expand_train_horizons,
        add_horizon_to_validation
    )
    from ..models_utils import predict_model, compute_metrics
    
    train_df, val_df, cov_df = load_data()

    # 1) Estatísticas do treino (antes de construir features)
    stats = compute_train_stats(train_df, cov_df)

    # 2) Features exógenas sobre treino + validação (uma única vez)
    exog_feats = build_exog_features(cov_df, stats)

    # 3) Bases de treino e validação
    X_train_df, y_train_std = build_train_base_direct(train_df, exog_feats, stats)
    X_validation = build_validation_base_direct(exog_feats, y_train_std, stats, VAL_DATES)
    
    X_train_expanded_df, y_train_expanded = expand_train_horizons(X_train_df, y_train_std)
    X_validation_horizon = add_horizon_to_validation(X_validation)
    
    model = DirectMLPForecaster(params=hyperparameters, horizons=28, n_jobs=1)
    
    model.fit(X_train_expanded_df, y_train_expanded)
    y_hat = predict_model(X_validation_horizon, stats, model, X_validation_horizon.columns, True)
    
    y_true = val_df.set_index("date").loc[VAL_DATES, SERIE].to_numpy()
    print(compute_metrics(y_true, y_hat.to_numpy(), train_df[SERIE].to_numpy()))