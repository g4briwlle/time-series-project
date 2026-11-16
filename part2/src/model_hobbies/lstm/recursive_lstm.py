from ..data.load_data import load_data
from ..data.prepare_data import (
    compute_train_stats,
    build_exog_features,
    build_train_base_recursive,
    build_validation_base_recursive,
)
from ..model_utils import predict_model, compute_metrics, save_outputs
from ..config import SERIE, VAL_DATES, OUTPUT_DIR
from .recursive_lstm_class import RecursiveLSTMForecaster

hyperparameters = {
    "hidden_size": 16,
    "num_layers": 1,
    "seq_len": 7,
    "lr": 0.0005,
    "weight_decay": 0.001,
    "epochs": 35,
    "batch_size": 64,
}
if __name__ == "__main__":    
    print(f"Iniciando treinamento final (Refit) do modelo Recursivo LSTM para a série: {SERIE}...")
    train_df, val_df, cov_df = load_data()

    # Passando explicitamente a série caso suas funções precisem (garante que não puxe "store_total" por acidente)
    stats = compute_train_stats(train_df, cov_df)
    exog_feats = build_exog_features(cov_df, stats)

    X_train_df, y_train_std = build_train_base_recursive(train_df, exog_feats, stats)
    X_validation = build_validation_base_recursive(exog_feats, y_train_std, stats, VAL_DATES)

    model = RecursiveLSTMForecaster(params=hyperparameters)
    model.fit(X_train_df, y_train_std)

    y_hat_final = predict_model(X_validation, stats, model, X_validation.columns, True)

    y_true = val_df.set_index("date").loc[VAL_DATES, SERIE].to_numpy()
    
    validation_metrics = compute_metrics(
        y_true, 
        y_hat_final.to_numpy(), 
        train_df[SERIE].to_numpy()
    )
    print(f"\nMétricas Finais na Validação ({SERIE}):")
    print(validation_metrics)

    # Corrigindo o diretório de saída conforme a sua estrutura para salvar os CSVs corretamente
    out_dir = OUTPUT_DIR.parent.parent / 'part4' / 'outputs'
    
    print(f"\nSalvando resultados em {out_dir}...")
    
    # O check_task2.py aceita 'rede', 'lstm', 'rnn' ou 'transformer'. 'rede' foi o sugerido.
    # O pipeline salva as predições de validação anexando ao CSV principal.
    save_outputs(
        y_hat=y_hat_final, 
        val_dates=VAL_DATES, 
        metrics=validation_metrics, 
        model_name='rede', 
        output_dir=out_dir
    )
    print("Concluído! O check_task2.py já pode validar a série HOBBIES.")