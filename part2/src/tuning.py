# tuning.py
"""
Nested Cross-Validation (2NCV) para seleção de hiperparâmetros.
- Outer: 2 folds (TimeSeriesSplit) — estimativa não enviesada.
- Inner: 3 folds (TimeSeriesSplit) — escolha de hiperparâmetros.
- Nunca toca a validação (2016-03-28 a 2016-04-24).
"""
import numpy as np
from itertools import product
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


def _expand_grid(param_grid):
    keys = list(param_grid.keys())
    return [dict(zip(keys, c)) for c in product(*[param_grid[k] for k in keys])]


def make_ridge(**p):  return Ridge(**p)
def make_rf(**p):     return RandomForestRegressor(random_state=42, n_jobs=-1, **p)
def make_hgb(**p):    return HistGradientBoostingRegressor(random_state=42, verbose=0, **p)


def nested_cv(X, y, model_factory, param_grid,
              outer_splits=2, inner_splits=3, metric=mean_absolute_error,
              nome="modelo"):
    """2NCV: inner escolhe params, outer dá estimativa não enviesada."""
    combos = _expand_grid(param_grid)
    outer = TimeSeriesSplit(n_splits=outer_splits)
    inner = TimeSeriesSplit(n_splits=inner_splits)

    outer_scores = []
    params_por_fold = []
    inner_means_por_combo = {i: [] for i in range(len(combos))}

    print(f"  [{nome}] 2NCV: {len(combos)} combos | outer={outer_splits} | inner={inner_splits}")
    for fi, (tr, te) in enumerate(outer.split(X), start=1):
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]

        inner_means = []
        for ci, p in enumerate(combos):
            scores = []
            for tr2, va2 in inner.split(X_tr):
                m = model_factory(**p)
                m.fit(X_tr[tr2], y_tr[tr2])
                scores.append(metric(y_tr[va2], m.predict(X_tr[va2])))
            mean_score = float(np.mean(scores))
            inner_means.append(mean_score)
            inner_means_por_combo[ci].append(mean_score)

        bi = int(np.argmin(inner_means))
        best_p = combos[bi]
        params_por_fold.append(best_p)

        m = model_factory(**best_p)
        m.fit(X_tr, y_tr)
        s = float(metric(y_te, m.predict(X_te)))
        outer_scores.append(s)
        print(f"    outer {fi}/{outer_splits}: MAE={s:.4f} | {best_p}")

    final_ci = min(inner_means_por_combo, key=lambda i: np.mean(inner_means_por_combo[i]))
    final_params = combos[final_ci]
    relatorio = {
        "outer_scores": outer_scores,
        "outer_mean": float(np.mean(outer_scores)),
        "params_por_fold": params_por_fold,
        "best_params_final": final_params,
        "inner_mean_best": float(np.mean(inner_means_por_combo[final_ci])),
    }
    print(f"    → params finais: {final_params} (inner MAE médio={relatorio['inner_mean_best']:.4f})")
    return final_params, relatorio