"""
Global LightGBM forecaster (M5-style): one L1 (median) gradient-boosted model trained across a
panel of per-series-mean-normalized weekly series, with lag/rolling features PLUS known
covariates (price, SNAP, events, calendar, dept/cat/state), and per-series split-conformal 80%
prediction intervals. A robust median GBM + conformal interval (rather than quantile GBMs, which
under-cover on short weekly series) gives a calibrated interval and a competitive point forecast.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..core.registry import FORECASTERS
from ..core.types import BaseForecast
from .base import clamp_nonneg, horizon

_LAGS = (1, 2, 3, 4, 8, 13, 26, 52)
_ROLL_MEAN = (4, 8, 13)
_ROLL_STD = (4, 8)


def _series_scale(a: np.ndarray) -> float:
    pos = a[a > 0]
    s = float(pos.mean()) if pos.size else 1.0
    return s if s > 0 else 1.0


# Exogenous (known-covariate) feature columns supplied by the data source.
_EXOG_COLS = ["log_price", "snap", "event", "month", "woy", "dept", "cat", "state"]


def _feat_row_exog(y: np.ndarray, exog: np.ndarray, t: int, fill: float) -> list[float]:
    """y-lag/rolling features (from normalized y) + known exog covariates at ``t``."""
    f = [y[t - lag] if t - lag >= 0 else fill for lag in _LAGS]
    for w in _ROLL_MEAN:
        win = y[max(0, t - w):t]
        f.append(float(win.mean()) if win.size else fill)
    for w in _ROLL_STD:
        win = y[max(0, t - w):t]
        f.append(float(win.std()) if win.size > 1 else 0.0)
    f.extend(exog[t].tolist())            # covariates are known for all weeks
    return f


def _recursive_exog(model, y_hist, exog: np.ndarray, fill: float, h: int, start: int) -> np.ndarray:
    """Recursively forecast positions ``start..start+h-1`` (exog known, y-lags fed back)."""
    running = list(y_hist)
    out = []
    for i in range(h):
        t = start + i
        row = np.array([_feat_row_exog(np.asarray(running), exog, t, fill)], dtype=float)
        pred = clamp_nonneg(float(model.predict(row)[0]))
        running.append(pred)
        out.append(pred)
    return np.array(out)


@FORECASTERS.register("lightgbm_global")
class GlobalLightGBMForecaster:
    """Global LightGBM the M5 way: one L1 model trained across a panel of scaled
    series with lag/rolling features PLUS known covariates (price, SNAP, events,
    calendar, dept/cat/state), and per-series conformal 80% intervals.

    Constructed with the training ``panel`` (item_id -> weekly series), the
    matching ``exog`` frames, and a ``train_cutoff`` week (only pre-cutoff data is
    used to fit — no leakage). Values are normalized per series by their in-sample
    mean. Fitted lazily and cached.
    """

    name = "lightgbm_global"
    REQUIRES = ("panel", "exog", "train_cutoff")   # capability: harness injects panel-training context

    def __init__(self, panel: dict | None = None, exog: dict | None = None,
                 train_cutoff: int | None = None, n_estimators: int = 500,
                 learning_rate: float = 0.05, num_leaves: int = 63,
                 min_child_samples: int = 20):
        self.panel = panel or {}
        self.exog = exog or {}
        self.train_cutoff = train_cutoff
        self.params = dict(
            objective="regression_l1", n_estimators=n_estimators,
            learning_rate=learning_rate, num_leaves=num_leaves,
            min_child_samples=min_child_samples, verbosity=-1,
        )
        self._model = None

    def _exog_arr(self, item_id: str, n: int) -> np.ndarray:
        df = self.exog.get(item_id)
        if df is None:
            return np.zeros((n, len(_EXOG_COLS)))
        return df[_EXOG_COLS].to_numpy(dtype=float)

    def _fit(self):
        import lightgbm as lgb

        cutoff = self.train_cutoff
        X_all, y_all = [], []
        for uid, s in self.panel.items():
            a = s.to_numpy(dtype=float)
            a = a[:cutoff] if cutoff else a
            if len(a) < 60:
                continue
            an = a / _series_scale(a)
            ex = self._exog_arr(uid, len(a))
            fill = float(an.mean())
            for t in range(1, len(an)):
                X_all.append(_feat_row_exog(an, ex, t, fill))
                y_all.append(an[t])
        if not X_all:
            raise ValueError("empty training panel for global LightGBM")
        self._model = lgb.LGBMRegressor(**self.params).fit(np.array(X_all), np.array(y_all))

    def predict(self, item_id: str, history: pd.Series, target_weeks: Sequence[int]) -> list[BaseForecast]:
        if self._model is None:
            self._fit()
        a = history.to_numpy(dtype=float)
        scale = _series_scale(a)
        an = a / scale
        fill = float(an.mean())
        ex = self._exog_arr(item_id, max(len(an), int(max(target_weeks)) + 1))
        h = horizon(history, target_weeks)

        # Per-series conformal interval offsets (normalized) from a validation tail.
        q_lo, q_hi = -0.4, 0.4
        split = int(len(an) * 0.85)
        if len(an) - split >= 4:
            preds = _recursive_exog(self._model, an[:split], ex, fill, len(an) - split, split)
            resid = an[split:] - preds
            q_lo, q_hi = np.percentile(resid, [10, 90])

        pt = _recursive_exog(self._model, an, ex, fill, h, len(an))
        last = int(history.index[-1])
        by_pos = {
            last + 1 + i: BaseForecast(
                point=clamp_nonneg(pt[i] * scale),
                lower=clamp_nonneg((pt[i] + q_lo) * scale),
                upper=clamp_nonneg((pt[i] + q_hi) * scale),
            )
            for i in range(h)
        }
        return [by_pos[int(w)] for w in target_weeks]
