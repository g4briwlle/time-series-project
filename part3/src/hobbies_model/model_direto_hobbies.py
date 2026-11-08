# model_direto_hobbies.py
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from .tuning import nested_cv, make_rf, make_hgb
except ImportError:
    from tuning import nested_cv, make_rf, make_hgb

DATA_DIR = Path("dados")
SERIE = "HOBBIES"
PRECO_COL = "preco_HOBBIES"
ORIGEM = pd.Timestamp("2016-03-27")
VAL_START = pd.Timestamp("2016-03-28")
VAL_END = pd.Timestamp("2016-04-24")
HORIZONTE = 28
LAGS = [1, 7, 14, 21, 28]
TIPOS_EVENTO = ["Sporting", "Cultural", "National", "Religious"]
GRID_RF = {"n_estimators": [300], "max_depth": [8, 12, None], "min_samples_leaf": [3, 5, 10]}
GRID_HGB = {"max_iter": [300], "learning_rate": [0.03, 0.1], "max_depth": [4, 6, None], "min_samples_leaf": [10, 20]}


def carregar_dados():
    train = pd.read_csv(DATA_DIR / "treino.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    val = pd.read_csv(DATA_DIR / "validacao.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    cov = pd.read_csv(DATA_DIR / "covariaveis.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return train, val, cov


def preparar_covariaveis(cov):
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
    return df


def _features_origem(row):
    f = {f"lag_{k}": row[f"lag_{k}"] for k in LAGS}
    f["roll_7"] = row["roll_7"]; f["roll_28"] = row["roll_28"]; f["y_origem"] = row[SERIE]
    return f


def _features_alvo(alvo, h):
    f = {"h": h, "dow": alvo["dow"], "day_of_year": alvo["day_of_year"],
         "sin_anual": alvo["sin_anual"], "cos_anual": alvo["cos_anual"],
         "preco": alvo[PRECO_COL], "snap": alvo["snap"], "is_event": alvo["is_event"]}
    for tipo in TIPOS_EVENTO:
        f[f"ev1_{tipo}"] = alvo[f"ev1_{tipo}"]; f[f"ev2_{tipo}"] = alvo[f"ev2_{tipo}"]
    return f


def construir_dataset(train, cov):
    df = train[["date", SERIE]].merge(cov, on="date", how="left").sort_values("date").reset_index(drop=True)
    for lag in LAGS: df[f"lag_{lag}"] = df[SERIE].shift(lag)
    df["roll_7"] = df[SERIE].shift(1).rolling(7).mean()
    df["roll_28"] = df[SERIE].shift(1).rolling(28).mean()
    datas = df["date"].values; idx = {d: i for i, d in enumerate(datas)}
    max_origem = datas[-1] - np.timedelta64(HORIZONTE, "D")
    regs = []
    for i, t in enumerate(datas):
        if t > max_origem: break
        base = _features_origem(df.iloc[i])
        for h in range(1, HORIZONTE + 1):
            j = idx.get(t + np.timedelta64(h, "D"))
            if j is None: continue
            alvo = df.iloc[j]
            if pd.isna(alvo[SERIE]): continue
            feat = base.copy(); feat.update(_features_alvo(alvo, h))
            feat["y"] = alvo[SERIE]; regs.append(feat)
    out = pd.DataFrame(regs)
    feature_cols = [c for c in out.columns if c != "y"]
    out = out.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)
    return out[feature_cols].to_numpy(), out["y"].to_numpy(), feature_cols, out, df


def prever_direto(modelo, feature_cols, df, cov):
    row = df[df["date"] == ORIGEM].iloc[0]
    base = _features_origem(row)
    datas_alvo = pd.date_range(VAL_START, VAL_END, freq="D")
    cov_alvo = cov.set_index("date").loc[datas_alvo]
    regs = []
    for h, data_alvo in enumerate(datas_alvo, start=1):
        feat = base.copy(); feat.update(_features_alvo(cov_alvo.loc[data_alvo], h))
        regs.append(feat)
    X = pd.DataFrame(regs)[feature_cols]
    return np.maximum(0.0, modelo.predict(X)), datas_alvo


def calcular_metricas(y_true, yhat, y_train):
    err = yhat - y_true
    mae = float(np.abs(err).mean()); rmse = float(np.sqrt((err ** 2).mean()))
    escala = float(np.abs(y_train[7:] - y_train[:-7]).mean())
    return {"mae": mae, "rmse": rmse, "mase": mae / escala if escala > 0 else float("nan")}


def main():
    print("--- DIRETO HOBBIES (com 2NCV) ---")
    train, val, cov = carregar_dados()
    cov = preparar_covariaveis(cov)
    X, y, feature_cols, df_full, df_lag = construir_dataset(train, cov)
    print(f"Dataset: {len(X)} linhas | {len(feature_cols)} features")

    y_true = val.set_index("date").loc[pd.date_range(VAL_START, VAL_END, freq="D"), SERIE].to_numpy()
    y_train_raw = train[SERIE].to_numpy()
    modelos = {"arvore": (make_rf, GRID_RF), "boosting": (make_hgb, GRID_HGB)}
    prevs, mets = [], []

    for nome, (factory, grid) in modelos.items():
        best_params, _ = nested_cv(X, y, factory, grid, nome=nome)
        modelo = factory(**best_params); modelo.fit(X, y)
        yhat, datas = prever_direto(modelo, feature_cols, df_lag, cov)
        m = calcular_metricas(y_true, yhat, y_train_raw)
        print(f"  [{nome}] VALIDAÇÃO: MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} MASE={m['mase']:.4f}")
        prevs.append(pd.DataFrame({"date": datas.strftime("%Y-%m-%d"), "series": SERIE, "modelo": nome, "yhat": yhat}))
        mets.append(pd.DataFrame([{"series": SERIE, "modelo": nome, **m}]))

    pd.concat(prevs).to_csv(f"previsoes_direto_{SERIE}.csv", index=False)
    pd.concat(mets).to_csv(f"metricas_direto_{SERIE}.csv", index=False)


if __name__ == "__main__":
    main()