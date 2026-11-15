from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPRegressor


class DirectMLPForecaster(BaseEstimator):
    """
    Estratégia direta com 28 MLPRegressor independentes (um por horizonte).

    Parameters
    ----------
    params : dict | None
        Kwargs repassados a cada sklearn.neural_network.MLPRegressor.
        Ex.: {"hidden_layer_sizes": (32,), "alpha": 1e-3,
              "learning_rate_init": 1e-3, "early_stopping": False}
        É este dict que você troca na CV.
    horizons : int
        Número de horizontes (28 para a Task 2).
    h_col : str
        Nome da coluna que identifica o horizonte em X. Removida das features.
    random_state : int
        Semente base; o modelo do horizonte h usa random_state + h para que
        os 28 modelos não sejam idênticos em inicialização.
    n_jobs : int
        Paralelismo do joblib ao ajustar os 28 modelos (-1 = todos os cores).

    Notes
    -----
    - Não use `h` como feature: ela é só coluna de despacho.
    - `early_stopping=True` dentro do MLPRegressor usa um split interno do
      *treino* (não da validação), então não é vazamento — mas com ~1900
      pontos e 28 modelos aumenta a variância. Deixei `early_stopping=False`
      como default no `_make_model`; sobrescreva via `params` se quiser.
    """

    def __init__(self, params: dict | None = None, horizons: int = 28,
                 h_col: str = "h", random_state: int = 42, n_jobs: int = -1):
        self.params = params
        self.horizons = horizons
        self.h_col = h_col
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------ #
    # construção
    # ------------------------------------------------------------------ #
    def _make_model(self, h: int) -> MLPRegressor:
        p = dict(self.params or {})
        p.setdefault("random_state", self.random_state + h)
        p.setdefault("max_iter", 1000)
        p.setdefault("early_stopping", False)   # evita variância extra
        return MLPRegressor(**p)

    # ------------------------------------------------------------------ #
    # fit
    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        """
        X : DataFrame (n, p) com coluna self.h_col  OU  (n, p) sem h se y for (n, 28)
        y : Series/array (n,) quando X tem h  OU  array/DataFrame (n, 28)
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.h_col in X.columns:
            self.feature_names_ = [c for c in X.columns if c != self.h_col]
            y_arr = np.asarray(y).ravel()
            if len(y_arr) != len(X):
                raise ValueError("y deve ter o mesmo comprimento de X (com h_col).")

            hs = sorted(X[self.h_col].astype(int).unique())
            Xn = X[self.feature_names_].to_numpy()
            h_vals = X[self.h_col].astype(int).to_numpy()

            def _fit_one(h):
                mask = h_vals == h
                m = self._make_model(h)
                m.fit(Xn[mask], y_arr[mask])
                return h, m

            self.models_ = dict(Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_one)(h) for h in hs
            ))
            # checagem de sanidade
            missing = set(range(1, self.horizons + 1)) - set(self.models_)
            if missing:
                raise ValueError(f"Horizontes ausentes no treino: {sorted(missing)}")
            return self

        # ---- fallback: X sem h, y com 28 colunas ----
        y_arr = np.asarray(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.horizons:
            raise ValueError(
                "Sem coluna h em X, y deve ser (n, horizons). "
                "Caso contrário use fit_horizon(h, X_h, y_h)."
            )
        self.feature_names_ = list(X.columns)
        Xn = X.to_numpy()

        def _fit_one(i):
            h = i + 1
            m = self._make_model(h)
            m.fit(Xn, y_arr[:, i])
            return h, m

        self.models_ = dict(Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(i) for i in range(self.horizons)
        ))
        return self

    def fit_horizon(self, h: int, X_h, y_h):
        """Ajusta só o modelo do horizonte h (útil se seu builder for por h)."""
        if not isinstance(X_h, pd.DataFrame):
            X_h = pd.DataFrame(X_h)
        h = int(h)
        cols = [c for c in X_h.columns if c != self.h_col]
        if getattr(self, "feature_names_", None) is None:
            self.feature_names_ = cols
        m = self._make_model(h)
        m.fit(X_h[self.feature_names_].to_numpy(), np.asarray(y_h).ravel())
        if not hasattr(self, "models_"):
            self.models_ = {}
        self.models_[h] = m
        return self

    # ------------------------------------------------------------------ #
    # predict
    # ------------------------------------------------------------------ #
    def _cols(self, X: pd.DataFrame) -> list[str]:
        if getattr(self, "feature_names_", None) is None:
            return [c for c in X.columns if c != self.h_col]
        return self.feature_names_

    def predict(self, X):
        """
        - X com coluna h  -> despacha para o modelo de cada h e devolve
          array na MESMA ordem das linhas de X.
        - X sem h e só um modelo ajustado -> usa esse modelo.
        - X sem h e vários modelos -> erro; use predict_at(h, X).
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        cols = self._cols(X)
        self._check_columns(X, cols)

        if self.h_col in X.columns:
            out = np.empty(len(X), dtype=float)
            h_vals = X[self.h_col].astype(int).to_numpy()
            Xn = X[cols].to_numpy()
            for h, m in self.models_.items():
                mask = h_vals == h
                if mask.any():
                    out[mask] = m.predict(Xn[mask])
            if np.isnan(out).any():
                missing = sorted(set(h_vals[np.isnan(out)].tolist()))
                raise ValueError(f"Sem modelo para horizonte(s): {missing}")
            return out

        if len(self.models_) == 1:
            return next(iter(self.models_.values())).predict(X[cols].to_numpy())
        raise ValueError(
            "X não tem coluna h e há 28 modelos ajustados. "
            "Use predict_at(h, X_h) ou inclua a coluna h."
        )

    def predict_at(self, h: int, X_h):
        """Predição explícita para um horizonte."""
        if not isinstance(X_h, pd.DataFrame):
            X_h = pd.DataFrame(X_h)
        cols = self._cols(X_h)
        self._check_columns(X_h, cols)
        return self.models_[int(h)].predict(X_h[cols].to_numpy())

    # ------------------------------------------------------------------ #
    # utilidades
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_columns(X: pd.DataFrame, cols: list[str]):
        pass
        # missing = [c for c in cols if c not in X.columns]
        # if missing:
        #     raise KeyError(f"Colunas ausentes em X: {missing}")

    @property
    def model(self) -> dict[int, MLPRegressor]:
        """Compat: seu snippet chama `model.predict(...)`. Como a classe já é
        o modelo, este alias é opcional — só para não quebrar nada."""
        return self.models_