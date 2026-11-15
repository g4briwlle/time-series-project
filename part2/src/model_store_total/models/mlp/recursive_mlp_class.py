from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPRegressor


class RecursiveMLPForecaster(BaseEstimator):
    """
    MLP com estratégia recursiva.

    Um único MLPRegressor ajustado a features recursivas (lags de y +
    janelas móveis + exógenas). Na predição, percorre a validação linha a
    linha, atualizando lags/rollings com o histórico de y padronizado
    (treino + previsões anteriores) e usando o modelo para prever o
    próximo passo. Devolve ŷ PADRONIZADO — compatível com `predict_model`
    do seu pipeline.

    Parameters
    ----------
    params : dict | None
        Kwargs repassados a MLPRegressor (mesmo formato do DirectMLPForecaster).
    lags : tuple[int]
        Lags de y que compõem features. Os nomes em X devem ser f"lag_{k}".
    rolling : tuple[(name, window, stat)]
        Features de janela móvel sobre o histórico padronizado.
        stat ∈ {"mean", "std"} (std usa ddof=1, igual à sua base).
    random_state : int
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

    # ------------------------------------------------------------------ #
    # construção / fit
    # ------------------------------------------------------------------ #
    def _make_model(self) -> MLPRegressor:
        p = dict(self.params or {})
        p.setdefault("random_state", self.random_state)
        p.setdefault("max_iter", 1000)
        p.setdefault("early_stopping", False)   # mesmo critério do direct
        return MLPRegressor(**p)

    def _min_history(self) -> int:
        return max(list(self.lags) + [w for _, w, _ in self.rolling])

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_cols_ = list(X.columns)
        y_arr = np.asarray(y).ravel()
        if len(y_arr) != len(X):
            raise ValueError("X e y com tamanhos diferentes.")
        if len(y_arr) < self._min_history():
            raise ValueError(
                f"y tem {len(y_arr)} pontos; features recursivas exigem "
                f"ao menos {self._min_history()}."
            )

        # Conferência: as colunas de lag/rolling prometidas existem em X.
        for k in self.lags:
            if f"lag_{k}" not in X.columns:
                raise KeyError(f"coluna lag_{k} ausente no X de treino")
        for name, _, _ in self.rolling:
            if name not in X.columns:
                raise KeyError(f"coluna {name} ausente no X de treino")

        self.model_ = self._make_model()
        self.model_.fit(X[self.feature_cols_].to_numpy(dtype=float), y_arr)

        # Semente do histórico: y_train PADRONIZADO, em ordem temporal.
        # É lido só para preencher os primeiros lags/rollings da recursão.
        self.history_seed_ = y_arr.astype(float).copy()
        return self

    # ------------------------------------------------------------------ #
    # predict
    # ------------------------------------------------------------------ #
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
                raise ValueError(f"stat desconhecido: {stat}")
        return row

    def predict(self, X) -> np.ndarray:
        """
        Recursão sobre X (uma linha por data do horizonte, em ordem temporal).

        Retorna ŷ padronizado, pronto para `predict_model` despadronizar.
        Para já sair na escala original, use `predict_raw`.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        missing = [c for c in self.feature_cols_ if c not in X.columns]
        if missing:
            raise KeyError(f"Colunas do treino ausentes em X: {missing}")

        history: list[float] = list(self.history_seed_)
        preds = np.empty(len(X), dtype=float)

        for i in range(len(X)):
            row = self._update_row(X.iloc[i], history)
            x_i = row[self.feature_cols_].to_numpy(dtype=float).reshape(1, -1)
            yhat_i = float(self.model_.predict(x_i)[0])
            preds[i] = yhat_i
            history.append(yhat_i)

        return preds

    def predict_raw(self, X, y_mean: float, y_std: float) -> np.ndarray:
        """Atalho: predict + despadronização com (y_mean, y_std)."""
        return self.predict(X) * y_std + y_mean