from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator

from .lstm_net import make_sliding_windows, train_torch_lstm


class RecursiveLSTMForecaster(BaseEstimator):
    """
    LSTM com estratégia recursiva (compatível com RecursiveMLPForecaster).

    Mantém histórico padronizado de y para atualizar lags/rollings a cada passo
    e mantém um buffer das últimas (seq_len - 1) linhas de features completas
    para montar o tensor 3D (1, seq_len, n_features) em cada dia do horizonte.
    """

    def __init__(
        self,
        params: dict | None = None,
        lags: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 14, 21, 28),
        rolling: tuple[tuple[str, int, str], ...] = (
            ("rolling_mean_7", 7, "mean"),
            ("rolling_mean_28", 28, "mean"),
            ("rolling_std_7", 7, "std"),
        ),
        random_state: int = 42,
    ):
        self.params = params
        self.lags = tuple(lags)
        self.rolling = tuple(rolling)
        self.random_state = random_state

    def _get_hparams(self) -> dict:
        p = dict(self.params or {})
        return {
            "hidden_size": p.get("hidden_size", 32),
            "num_layers": p.get("num_layers", 1),
            "seq_len": p.get("seq_len", 7),
            "dropout": p.get("dropout", 0.0),
            "lr": p.get("lr", 1e-3),
            "weight_decay": p.get("weight_decay", 1e-4),
            "epochs": p.get("epochs", 40),
            "batch_size": p.get("batch_size", 64),
        }

    def _min_history(self) -> int:
        return max(list(self.lags) + [w for _, w, _ in self.rolling])

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_cols_ = list(X.columns)
        y_arr = np.asarray(y, dtype=np.float32).ravel()

        if len(y_arr) != len(X):
            raise ValueError("X e y têm comprimentos diferentes.")
        if len(y_arr) < self._min_history():
            raise ValueError(
                f"y tem {len(y_arr)} pontos; mínimo exigido é {self._min_history()}."
            )

        for k in self.lags:
            if f"lag_{k}" not in X.columns:
                raise KeyError(f"Coluna lag_{k} ausente no X de treino.")
        for name, _, _ in self.rolling:
            if name not in X.columns:
                raise KeyError(f"Coluna {name} ausente no X de treino.")

        hp = self._get_hparams()
        self.seq_len_ = hp["seq_len"]

        X_2d = X[self.feature_cols_].to_numpy(dtype=np.float32)
        X_3d, y_3d = make_sliding_windows(X_2d, y_arr, seq_len=self.seq_len_)

        self.model_ = train_torch_lstm(
            X_3d,
            y_3d,
            hidden_size=hp["hidden_size"],
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
            lr=hp["lr"],
            weight_decay=hp["weight_decay"],
            epochs=hp["epochs"],
            batch_size=hp["batch_size"],
            seed=self.random_state,
        )

        # Sementes para a recursão na validação (apenas dados do treino!)
        self.history_seed_ = y_arr.astype(float).copy()
        self.feature_buffer_seed_ = (
            X_2d[-(self.seq_len_ - 1) :].copy()
            if self.seq_len_ > 1
            else np.empty((0, X_2d.shape[1]), dtype=np.float32)
        )
        return self

    def _update_row(self, row: pd.Series, history: list[float]) -> pd.Series:
        row = row.copy()
        for k in self.lags:
            row[f"lag_{k}"] = history[-k]
        for name, w, stat in self.rolling:
            win = np.asarray(history[-w:], dtype=float)
            if stat == "mean":
                row[name] = float(win.mean())
            elif stat == "std":
                row[name] = float(win.std(ddof=1))
            else:
                raise ValueError(f"Estatística desconhecida: {stat}")
        return row

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        missing = [c for c in self.feature_cols_ if c not in X.columns]
        if missing:
            raise KeyError(f"Colunas ausentes em X: {missing}")

        history: list[float] = list(self.history_seed_)
        feat_buffer: list[np.ndarray] = [
            row for row in self.feature_buffer_seed_
        ]
        preds = np.empty(len(X), dtype=float)

        self.model_.eval()
        with torch.no_grad():
            for i in range(len(X)):
                row = self._update_row(X.iloc[i], history)
                x_vec = row[self.feature_cols_].to_numpy(dtype=np.float32)
                feat_buffer.append(x_vec)

                # Monta janela (1, seq_len, n_features) com os últimos seq_len dias
                window = np.stack(feat_buffer[-self.seq_len_ :], axis=0)[np.newaxis, :, :]
                tensor_x = torch.from_numpy(window)
                yhat_i = float(self.model_(tensor_x).item())

                preds[i] = yhat_i
                history.append(yhat_i)

        return preds