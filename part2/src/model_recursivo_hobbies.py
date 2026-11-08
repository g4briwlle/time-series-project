# model_linear_recursivo_hobbies.py
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from .tuning import nested_cv, make_ridge
except ImportError:
    from tuning import nested_cv, make_ridge

DATA_DIR = Path("dados")
SERIE = "HOBBIES"
PRECO_COL = "preco_HOBBIES"
ORIGEM = pd.Timestamp("2016-03-27")
VAL_START = pd.Timestamp("2016-03-28")
VAL_END = pd.Timestamp("2016-04-24")
LAGS = [1, 7, 14, 21, 28]
TIPOS_EVENTO = ["Sporting", "Cultural", "National", "Religious"]
GRID_RIDGE = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}


def carregar_dados():
    train = pd.read_csv(DATA_DIR / "treino.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    val = pd.read_csv(DATA_DIR / "validacao.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    cov = pd.read_csv(DATA_DIR / "covariaveis.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return train, val, cov


def preparar_covariaveis(cov, stats):
    df = cov.copy()
    df["is_event"] = (df["event_type_1"].notna() | df["event_type_2"].notna()).astype(int)
    for tipo in TIPOS_EVENTO:
        df[f"ev1_{tipo}"] = (df["event_type_1"] == tipo).astype(int)
        df[f"ev2_{tipo}"] = (df["event_type_2"] == tipo).astype(int)
    df["dow"] = df["date"].dt.day_of_week
    df["day_of_year"] = df["date"].dt.day_of_year
    df["sin_anual"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_anual"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["snap"] = df["snap_CA"]
    df["preco_std"] = (df[PRECO_COL] - stats["preco_mean"]) / stats["preco_std"]
    return df


def calcular_stats(train, cov):
    cov_train = cov[cov["date"] <= ORIGEM]
    return {"y_mean": float(train[SERIE].mean()), "y_std": float(train[SERIE].std(ddof=0)),
            "preco_mean": float(cov_train[PRECO_COL].mean()),
            "preco_std": float(cov_train[PRECO_COL].std(ddof=0))}


def _features_dia(t, y_hist_std, cov_row, feature_cols):
    f = {f"lag_{k}": y_hist_std[-k] for k in LAGS}
    f["roll_7"] = float(np.mean(y_hist_std[-7:]))
    f["roll_28"] = float(np.mean(y_hist_std[-28:]))
    f["preco_std"] = cov_row["preco_std"]; f["snap"] = cov_row["snap"]; f["is_event"] = cov_row["is_event"]
    for tipo in TIPOS_EVENTO:
        f[f"ev1_{tipo}"] = cov_row[f"ev1_{tipo}"]; f[f"ev2_{tipo}"] = cov_row[f"ev2_{tipo}"]
    f["dow"] = t.day_of_week; f["day_of_year"] = t.day_of_year
    f["sin_anual"] = np.sin(2 * np.pi * t.day_of_year / 365.25)
    f["cos_anual"] = np.cos(2 * np.pi * t.day_of_year / 365.25)
    return {c: f[c] for c in feature_cols}


def construir_dataset(train, cov, stats):
    df = train[["date", SERIE]].merge(cov, on="date", how="left").sort_values("date").reset_index(drop=True)
    df["y_std"] = (df[SERIE] - stats["y_mean"]) / stats["y_std"]
    for lag in LAGS: df[f"lag_{lag}"] = df["y_std"].shift(lag)
    df["roll_7"] = df["y_std"].shift(1).rolling(7).mean()
    df["roll_28"] = df["y_std"].shift(1).rolling(28).mean()
    feature_cols = ([f"lag_{k}" for k in LAGS] + ["roll_7", "roll_28", "preco_std", "snap", "is_event",
                    "dow", "day_of_year", "sin_anual", "cos_anual"]
                    + [f"ev1_{t}" for t in TIPOS_EVENTO] + [f"ev2_{t}" for t in TIPOS_EVENTO])
    out = df.dropna(subset=feature_cols + ["y_std"]).reset_index(drop=True)
    return out[feature_cols].to_numpy(), out["y_std"].to_numpy(), feature_cols


def prever_recursivo(modelo, train, cov, feature_cols, stats):
    y_hist_std = list((train[SERIE] - stats["y_mean"]) / stats["y_std"])
    datas_alvo = pd.date_range(VAL_START, VAL_END, freq="D")
    cov_alvo = cov.set_index("date").loc[datas_alvo]
    preds_std = []
    for data_alvo in datas_alvo:
        feat = _features_dia(data_alvo, y_hist_std, cov_alvo.loc[data_alvo], feature_cols)
        pred = float(modelo.predict(np.array([[feat[c] for c in feature_cols]]))[0])
        preds_std.append(pred); y_hist_std.append(pred)
    return np.maximum(0.0, np.array(preds_std) * stats["y_std"] + stats["y_mean"]), datas_alvo


def calcular_metricas(y_true, yhat, y_train):
    err = yhat - y_true
    mae = float(np.abs(err).mean()); rmse = float(np.sqrt((err ** 2).mean()))
    escala = float(np.abs(y_train[7:] - y_train[:-7]).mean())
    return {"mae": mae, "rmse": rmse, "mase": mae / escala if escala > 0 else float("nan")}


def main():
    print("--- LINEAR RECURSIVO HOBBIES (com 2NCV) ---")
    train, val, cov = carregar_dados()
    stats = calcular_stats(train, cov)
    cov = preparar_covariaveis(cov, stats)
    X, y, feature_cols = construir_dataset(train, cov, stats)
    print(f"Dataset: {len(X)} linhas | {len(feature_cols)} features")

    best_params, _ = nested_cv(X, y, make_ridge, GRID_RIDGE, nome="linear")
    modelo = make_ridge(**best_params); modelo.fit(X, y)
    yhat, datas = prever_recursivo(modelo, train, cov, feature_cols, stats)
    y_true = val.set_index("date").loc[datas, SERIE].to_numpy()
    m = calcular_metricas(y_true, yhat, train[SERIE].to_numpy())
    print(f"  [linear] VALIDAÇÃO: MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} MASE={m['mase']:.4f}")

    pd.DataFrame({"date": datas.strftime("%Y-%m-%d"), "series": SERIE, "modelo": "linear", "yhat": yhat}).to_csv(
        f"previsoes_linear_recursivo_{SERIE}.csv", index=False)
    pd.DataFrame([{"series": SERIE, "modelo": "linear", **m}]).to_csv(
        f"metricas_linear_recursivo_{SERIE}.csv", index=False)


if __name__ == "__main__":
    main()