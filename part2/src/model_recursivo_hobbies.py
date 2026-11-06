import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, List, Dict

# ==============================================================================
# CONFIGURAÇÕES GERAIS
# ==============================================================================
BASE_PATH = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_PATH / 'dados'
SERIE = "HOBBIES"
PRECO_COL = "preco_HOBBIES"
ORIGEM = pd.Timestamp("2016-03-27")
VAL_START = pd.Timestamp("2016-03-28")
VAL_END = pd.Timestamp("2016-04-24")
LAGS = [1, 7, 14, 21, 28]

def carregar_dados() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega os dados brutos e garante ordenação temporal."""
    train_df = pd.read_csv(DATA_DIR / "treino.csv", parse_dates=["date"]).sort_values("date")
    val_df = pd.read_csv(DATA_DIR / "validacao.csv", parse_dates=["date"]).sort_values("date")
    cov_df = pd.read_csv(DATA_DIR / "covariaveis.csv", parse_dates=["date"]).sort_values("date")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), cov_df.reset_index(drop=True)

def calcular_estatisticas_treino(train_df: pd.DataFrame, cov_df: pd.DataFrame) -> Dict[str, float]:
    """Calcula média e desvio padrão EXCLUSIVAMENTE com dados do treino (sem leakage)."""
    y_mean = train_df[SERIE].mean()
    y_std = train_df[SERIE].std(ddof=0)
    
    cov_train = cov_df[cov_df["date"] <= ORIGEM]
    preco_mean = cov_train[PRECO_COL].mean()
    preco_std = cov_train[PRECO_COL].std(ddof=0)
    
    return {"y_mean": y_mean, "y_std": y_std, "preco_mean": preco_mean, "preco_std": preco_std}

def preparar_exogenas(cov_df: pd.DataFrame, stats: Dict[str, float]) -> pd.DataFrame:
    """Gera covariáveis de calendário, eventos e preço padronizado."""
    df = cov_df[["date", PRECO_COL, "snap_CA", "event_type_1", "event_type_2"]].copy()
    
    df["preco_std"] = (df[PRECO_COL] - stats["preco_mean"]) / stats["preco_std"]
    df["is_event"] = (df["event_type_1"].notna() | df["event_type_2"].notna()).astype(float)
    df["is_national_or_religious"] = (
        df["event_type_1"].isin(["National", "Religious"]) | 
        df["event_type_2"].isin(["National", "Religious"])
    ).astype(float)
    
    dow = df["date"].dt.dayofweek
    for d in range(1, 7):
        df[f"dow_{d}"] = (dow == d).astype(float)
        
    day_of_year = df["date"].dt.dayofyear
    df["sin_anual"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["cos_anual"] = np.cos(2 * np.pi * day_of_year / 365.25)
    
    cols_drop = [PRECO_COL, "event_type_1", "event_type_2"]
    return df.drop(columns=cols_drop)

def construir_base_treino(train_df: pd.DataFrame, exog_df: pd.DataFrame, stats: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Une variáveis, padroniza o alvo e cria os lags/rolagens para treino."""
    df = pd.merge(train_df[["date", SERIE]], exog_df, on="date", how="left").sort_values("date")
    df["y_std"] = (df[SERIE] - stats["y_mean"]) / stats["y_std"]
    
    for lag in LAGS:
        df[f"lag_{lag}"] = df["y_std"].shift(lag)
        
    df["rolling_mean_7"] = df["y_std"].shift(1).rolling(7).mean()
    df["rolling_mean_28"] = df["y_std"].shift(1).rolling(28).mean()
    
    df_clean = df.dropna().reset_index(drop=True)
    features = [c for c in df_clean.columns if c not in ("date", SERIE, "y_std")]
    
    return df_clean[features].to_numpy(), df_clean["y_std"].to_numpy(), features

def treinar_modelo_linear(X_train: np.ndarray, y_train: np.ndarray) -> ElasticNet:
    """Busca os melhores hiperparâmetros com Validação Cruzada Temporal e retorna o modelo."""
    tscv = TimeSeriesSplit(n_splits=5, test_size=28)
    melhor_score = float("inf")
    melhor_cfg = (1.0, 0.0) # Fallback seguro
    
    for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
        for l1_ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
            erros_cv = []
            for tr_idx, te_idx in tscv.split(X_train):
                mod = Ridge(alpha=alpha) if l1_ratio == 0.0 else ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=42)
                mod.fit(X_train[tr_idx], y_train[tr_idx])
                erros_cv.append(mean_absolute_error(y_train[te_idx], mod.predict(X_train[te_idx])))
            
            media_cv = np.mean(erros_cv)
            if media_cv < melhor_score:
                melhor_score = media_cv
                melhor_cfg = (alpha, l1_ratio)
                
    best_alpha, best_l1 = melhor_cfg
    print(f"[{SERIE}] Melhor hiperparâmetro cv: alpha={best_alpha}, l1_ratio={best_l1}")
    
    modelo_final = Ridge(alpha=best_alpha) if best_l1 == 0.0 else ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=5000, random_state=42)
    modelo_final.fit(X_train, y_train)
    return modelo_final

def prever_validacao_recursiva(modelo, train_df: pd.DataFrame, exog_df: pd.DataFrame, features: List[str], stats: Dict[str, float]) -> np.ndarray:
    """Realiza a previsão recursiva passo a passo no horizonte de validação."""
    y_hist_std = list((train_df[SERIE] - stats["y_mean"]) / stats["y_std"])
    val_dates = pd.date_range(VAL_START, VAL_END, freq="D")
    exog_val = exog_df[exog_df["date"].isin(val_dates)].sort_values("date")
    
    yhat_std = []
    for _, row in exog_val.iterrows():
        feat_dict = {col: row[col] for col in exog_val.columns if col != "date"}
        for lag in LAGS:
            feat_dict[f"lag_{lag}"] = y_hist_std[-lag]
            
        feat_dict["rolling_mean_7"] = np.mean(y_hist_std[-7:])
        feat_dict["rolling_mean_28"] = np.mean(y_hist_std[-28:])
        
        X_step = np.array([[feat_dict[c] for c in features]])
        pred = modelo.predict(X_step)[0]
        
        yhat_std.append(pred)
        y_hist_std.append(pred)
        
    yhat = np.maximum(0.0, np.array(yhat_std) * stats["y_std"] + stats["y_mean"])
    return yhat

def calcular_metricas(y_true: np.ndarray, yhat: np.ndarray, y_train_raw: np.ndarray) -> Dict[str, float]:
    """Calcula MAE, RMSE e MASE."""
    err = yhat - y_true
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err**2).mean()))
    mase_scale = float(np.abs(y_train_raw[7:] - y_train_raw[:-7]).mean())
    mase = mae / mase_scale if mase_scale > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "mase": mase}

def main():
    print(f"--- Iniciando Pipeline Linear: {SERIE} ---")
    train_df, val_df, cov_df = carregar_dados()
    
    stats = calcular_estatisticas_treino(train_df, cov_df)
    exog_df = preparar_exogenas(cov_df, stats)
    X_train, y_train, features = construir_base_treino(train_df, exog_df, stats)
    
    modelo = treinar_modelo_linear(X_train, y_train)
    yhat = prever_validacao_recursiva(modelo, train_df, exog_df, features, stats)
    
    val_dates = pd.date_range(VAL_START, VAL_END, freq="D")
    y_true = val_df.set_index("date").loc[val_dates, SERIE].to_numpy()
    
    metricas = calcular_metricas(y_true, yhat, train_df[SERIE].to_numpy())
    print(f"Métricas {SERIE}: MAE={metricas['mae']:.4f} | RMSE={metricas['rmse']:.4f} | MASE={metricas['mase']:.4f}")
    
    # Exportação Parcial
    pd.DataFrame({
        "date": val_dates.strftime("%Y-%m-%d"), "series": SERIE, "modelo": "linear", "yhat": yhat
    }).to_csv(f"previsoes_linear_recursivo_{SERIE}.csv", index=False)
    
    pd.DataFrame([{
        "series": SERIE, "modelo": "linear", "mae": metricas['mae'], "rmse": metricas['rmse'], "mase": metricas['mase']
    }]).to_csv(f"metricas_linear_recursivo_{SERIE}.csv", index=False)
    print("Arquivos parciais (recursivo) gerados com sucesso!")

if __name__ == "__main__":
    main()