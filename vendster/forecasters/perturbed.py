"""Controlled perturbation-grid forecaster.

Synthesizes a forecast from the *true* base demand with known bias direction and
variance, decoupling the two failure modes the paper studies:

  point  = true_base * exp(s * bias_magnitude)             # s in {-1, 0, +1}; LOG-bias
  P10/P90 = point * exp(-/+ z * variance_mult * sigma)      # sigma from product CV

Bias is applied in log space (``exp(s*magnitude)``), so +m and -m are exact reciprocal
distortions (a fair over/under symmetry test); variance widens/narrows the interval.
Low-variance + biased is the "confident and wrong" cell.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

from ..core.registry import FORECASTERS
from ..core.types import BaseForecast
from .base import clamp_nonneg

_Z80 = 1.2815515594603558   # standard-normal 0.9 quantile -> 80% interval


@FORECASTERS.register("perturbed")
class PerturbedForecaster:
    """Oracle-with-controlled-error over the true series (ignores ``history``)."""

    name = "perturbed"
    REQUIRES = ("true_weekly", "sigmas")     # capability: harness injects the true series + noise

    def __init__(
        self,
        bias_direction: int = 0,
        bias_magnitude: float = 0.20,
        variance_mult: float = 0.25,
        true_weekly: dict | None = None,
        sigmas: dict | None = None,
    ):
        if bias_direction not in (-1, 0, 1):
            raise ValueError("bias_direction must be -1, 0, or 1")
        self.s = bias_direction
        self.bias_magnitude = bias_magnitude
        self.variance_mult = variance_mult
        self.true_weekly = true_weekly or {}
        self.sigmas = sigmas or {}
        self._bias_factor = self.s * self.bias_magnitude       # log-bias -> point = true*exp(this)

    def predict(self, item_id: str, history: pd.Series, target_weeks: Sequence[int]) -> list[BaseForecast]:
        series = self.true_weekly[item_id]
        sigma = self.sigmas.get(item_id, 0.3)
        half = _Z80 * self.variance_mult * sigma
        out = []
        for w in target_weeks:
            true = float(series.iloc[int(w)])
            point = true * math.exp(self._bias_factor)
            out.append(BaseForecast(
                point=clamp_nonneg(point),
                lower=clamp_nonneg(point * math.exp(-half)),
                upper=clamp_nonneg(point * math.exp(half)),
            ))
        return out
