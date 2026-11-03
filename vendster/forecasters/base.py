"""Shared helpers for forecasters that predict from an aligned weekly series.

``history`` is a ``pd.Series`` whose integer index gives absolute week positions
(0-based, aligned across all products). ``target_weeks`` are absolute positions
strictly after the end of ``history``. Forecasters forecast ``horizon`` steps and
return one :class:`BaseForecast` per requested target week, in that order.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def horizon(history: pd.Series, target_weeks: Sequence[int]) -> int:
    last = int(history.index[-1])
    h = int(max(target_weeks)) - last
    if h < 1:
        raise ValueError("target_weeks must lie strictly after history")
    return h


def clamp_nonneg(x: float) -> float:
    return float(max(0.0, x))
