"""Classical statistical forecasters via statsforecast (Seasonal Naive, AutoETS, AutoTheta)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..core.registry import FORECASTERS
from ..core.types import BaseForecast
from .base import clamp_nonneg, horizon

_SEASON = 52          # weekly data -> yearly seasonality
_LEVEL = 80           # 80% prediction interval -> lo-80 = P10, hi-80 = P90


class _StatsForecaster:
    """Base wrapper around a single statsforecast model with 80% intervals.

    ``_log_transform`` fits in log1p space and inverts with expm1 — variance
    stabilization for count/grocery series (turns additive dynamics multiplicative
    and yields asymmetric, non-negative intervals).
    """

    name = ""
    _model_alias = ""
    _log_transform = False

    def _make_model(self):
        raise NotImplementedError

    def predict(self, item_id: str, history: pd.Series, target_weeks: Sequence[int]) -> list[BaseForecast]:
        from statsforecast import StatsForecast

        h = horizon(history, target_weeks)
        y = history.to_numpy(dtype=float)
        if self._log_transform:
            y = np.log1p(y)
        df = pd.DataFrame({"unique_id": item_id, "ds": history.index.astype(int), "y": y})
        sf = StatsForecast(models=[self._make_model()], freq=1, n_jobs=1)
        fc = sf.forecast(df=df, h=h, level=[_LEVEL]).reset_index(drop=True)
        alias = self._model_alias

        def _inv(v: float) -> float:
            return clamp_nonneg(np.expm1(v) if self._log_transform else v)

        by_pos: dict[int, BaseForecast] = {}
        for _, r in fc.iterrows():
            pos = int(r["ds"])
            by_pos[pos] = BaseForecast(
                point=_inv(r[alias]),
                lower=_inv(r[f"{alias}-lo-{_LEVEL}"]),
                upper=_inv(r[f"{alias}-hi-{_LEVEL}"]),
            )
        return [by_pos[int(w)] for w in target_weeks]


@FORECASTERS.register("seasonal_naive")
class SeasonalNaiveForecaster:
    """Native seasonal naive: point = value one season ago; 80% interval from the
    empirical distribution of in-sample seasonal 1-step errors. No statsforecast
    overhead (~instant), so a full pool census is cheap.
    """

    name = "seasonal_naive"

    def __init__(self, season_length: int = _SEASON):
        self.m = season_length

    def predict(self, item_id: str, history: pd.Series, target_weeks: Sequence[int]) -> list[BaseForecast]:
        a = history.to_numpy(dtype=float)
        last = int(history.index[-1])
        if len(a) > self.m:
            resid = a[self.m:] - a[:-self.m]
            q_lo, q_hi = np.percentile(resid, [10, 90])
        else:
            q_lo, q_hi = 0.0, 0.0
        h = horizon(history, target_weeks)
        running = list(a)
        by_pos = {}
        for step in range(1, h + 1):
            w = last + step
            ref = w - self.m                      # position one season before w
            val = running[ref] if 0 <= ref < len(running) else float(a.mean())
            running.append(val)                   # seasonal recursion for long horizons
            by_pos[w] = BaseForecast(
                point=clamp_nonneg(val),
                lower=clamp_nonneg(val + q_lo),
                upper=clamp_nonneg(val + q_hi),
            )
        return [by_pos[int(w)] for w in target_weeks]


@FORECASTERS.register("auto_ets")
class AutoETSForecaster(_StatsForecaster):
    """Standard per-series AutoETS baseline (literature-standard; log1p available
    on the base class but did not improve accuracy on these series)."""

    name = "auto_ets"
    _model_alias = "AutoETS"

    def _make_model(self):
        from statsforecast.models import AutoETS
        return AutoETS(season_length=_SEASON)


@FORECASTERS.register("auto_theta")
class AutoThetaForecaster(_StatsForecaster):
    """AutoTheta — a classical baseline that typically beats seasonal naive
    (positive skill in the Chronos-2 benchmarks)."""

    name = "auto_theta"
    _model_alias = "AutoTheta"

    def _make_model(self):
        from statsforecast.models import AutoTheta
        return AutoTheta(season_length=_SEASON)


# NOTE: AutoARIMA was prototyped here but EXCLUDED from the study everywhere (census + episodes):
# it clusters with AutoTheta/AutoETS (both fully run) so adds no breadth, and refits per predict
# (~20x/episode) under the global setup lock (~2.9h), which stalls the run. See config.py.
