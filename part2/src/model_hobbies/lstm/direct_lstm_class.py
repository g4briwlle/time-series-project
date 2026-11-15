from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator

from .lstm_net import make_sliding_windows, train_torch_lstm


class DirectLSTMForecaster(BaseEstimator):
    """
    Estratégia direta com LSTM, 100% compatível com o contrato do DirectMLPForecaster.

    Parameters
    ----------
    params : dict | None
        Hiperparâmetros da LSTM:
        - hidden_size (int): default 32
        - num_layers (int): default 1
        - seq_len (int): tamanho da janela temporal (default 7)
        - dropout (float): default 0.0
        - lr (float): learning rate (default 1e-3)
        - weight_decay (float): regularização L2 / alpha (default 1e-4)
        - epochs (int): default 35
        - batch_size (int): default 64
        - mode (str): "pooled" (1 LSTM com h codificado) ou "independent" (28 LSTMs)
    horizons : int
        Número de horizontes (28 na Task 2).
    h_col : str
        Coluna identificadora do horizonte (default "h").
    random_state : int
        Semente para reprodutibilidade.
    n_jobs : int
        Paralelismo usado quando mode="independent".
    """

    def __init__(
        self,
        params: dict | None = None,
        horizons: int = 28,
        h_col: str = "h",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.params = params
        self.horizons = horizons
        self.h_col = h_col
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _get_hparams(self) -> dict:
        p = dict(self.params or {})
        return {
            "hidden_size": p.get("hidden_size", 32),
            "num_layers": p.get("num_layers", 1),
            "seq_len": p.get("seq_len", 7),
            "dropout": p.get("dropout", 0.0),
            "lr": p.get("lr", 1e-3),
            "weight_decay": p.get("weight_decay", 1e-4),
            "epochs": p.get("epochs", 35),
            "batch_size": p.get("batch_size", 64),
            "mode": p.get("mode", "pooled"),
        }

    def _encode_h(self, X_feat: np.ndarray, h_val: int) -> np.ndarray:
        """Adiciona codificação normalizada e cíclica (período 7) do horizonte h."""
        n = len(X_feat)
        h_norm = np.full((n, 1), h_val / float(self.horizons), dtype=np.float32)
        h_sin = np.full((n, 1), np.sin(2 * np.pi * h_val / 7.0), dtype=np.float32)
        h_cos = np.full((n, 1), np.cos(2 * np.pi * h_val / 7.0), dtype=np.float32)
        return np.hstack([X_feat, h_norm, h_sin, h_cos])

    # direct_lstm_class.py — additions to the class

    def _prepare_windows(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        seq_len: int,
        mode: str,
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
        """
        Build per-horizon 3-D windows once, so fit() can skip make_sliding_windows.
        Returns {h: (X_3d, y_3d, last_context)} where last_context is the last
        (seq_len - 1) encoded rows of that horizon (or None if seq_len <= 1).
        """
        h_vals = X[self.h_col].astype(int).to_numpy()
        feature_names = [c for c in X.columns if c != self.h_col]
        X_raw = X[feature_names].to_numpy(dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).ravel()

        out: dict[int, tuple] = {}
        for h in sorted(np.unique(h_vals)):
            mask = h_vals == h
            Xh = self._encode_h(X_raw[mask], h) if mode == "pooled" else X_raw[mask]
            yh = y_arr[mask]
            ctx = Xh[-(seq_len - 1):].copy() if seq_len > 1 else None
            X3d, y3d = make_sliding_windows(Xh, yh, seq_len=seq_len)
            out[int(h)] = (X3d, y3d, ctx)
        return out


    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        precomputed_windows: dict | None = None,
    ):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        hp = self._get_hparams()
        self.seq_len_ = hp["seq_len"]
        self.mode_ = hp["mode"]
        self.device_ = _resolve_device(self.device)

        if self.h_col not in X.columns:
            raise ValueError(f"Coluna '{self.h_col}' obrigatória em X.")

        self.feature_names_ = [c for c in X.columns if c != self.h_col]
        self.last_train_windows_ = {}

        # ── single change: reuse windows if supplied ────────────────────
        if precomputed_windows is None:
            windows = self._prepare_windows(X, y, self.seq_len_, self.mode_)
        else:
            windows = precomputed_windows
        # ────────────────────────────────────────────────────────────────

        if self.mode_ == "pooled":
            X_all_3d = np.concatenate([v[0] for v in windows.values()], axis=0)
            y_all    = np.concatenate([v[1] for v in windows.values()], axis=0)
            for h, (_, _, ctx) in windows.items():
                if ctx is not None:
                    self.last_train_windows_[h] = ctx

            self.pooled_model_ = train_torch_lstm(
                X_all_3d, y_all,
                hidden_size=hp["hidden_size"], num_layers=hp["num_layers"],
                dropout=hp["dropout"], lr=hp["lr"], weight_decay=hp["weight_decay"],
                epochs=hp["epochs"], batch_size=hp["batch_size"],
                seed=self.random_state, device=self.device_,
            ).to(self.device_)
            return self

        # independent mode
        def _fit_one(h, X3d, y3d):
            return h, train_torch_lstm(
                X3d, y3d,
                hidden_size=hp["hidden_size"], num_layers=hp["num_layers"],
                dropout=hp["dropout"], lr=hp["lr"], weight_decay=hp["weight_decay"],
                epochs=hp["epochs"], batch_size=hp["batch_size"],
                seed=self.random_state + h, device=self.device_,
            )

        n_jobs = 1 if self.device_.type == "cuda" else self.n_jobs
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one)(h, X3d, y3d) for h, (X3d, y3d, _) in windows.items()
        )
        self.models_ = {h: net.to(self.device_) for h, net in results}
        for h, (_, _, ctx) in windows.items():
            if ctx is not None:
                self.last_train_windows_[h] = ctx
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        h_vals = X[self.h_col].astype(int).to_numpy()
        X_raw = X[self.feature_names_].to_numpy(dtype=np.float32)
        out = np.empty(len(X), dtype=float)

        with torch.no_grad():
            for h in np.unique(h_vals):
                mask = h_vals == h
                Xh = self._encode_h(X_raw[mask], int(h)) if self.mode_ == "pooled" else X_raw[mask]

                # Concatena o final do treino daquele horizonte para formar a janela causal
                if self.seq_len_ > 1 and int(h) in self.last_train_windows_:
                    ctx = self.last_train_windows_[int(h)]
                    Xh_full = np.vstack([ctx, Xh])
                    X_3d, _ = make_sliding_windows(Xh_full, None, seq_len=self.seq_len_)
                    X_3d = X_3d[len(ctx) :]  # mantém apenas as linhas de X (validação)
                else:
                    X_3d, _ = make_sliding_windows(Xh, None, seq_len=self.seq_len_)

                tensor_x = torch.from_numpy(X_3d)
                net = self.pooled_model_ if self.mode_ == "pooled" else self.models_[int(h)]
                preds = net(tensor_x).numpy()
                out[mask] = preds

        return out