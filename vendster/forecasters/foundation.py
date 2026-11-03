"""Foundation-model forecasters (zero-shot probabilistic): Chronos-2, Chronos-Bolt, TimesFM-2.5."""

from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..core.registry import FORECASTERS
from ..core.types import BaseForecast
from .base import clamp_nonneg, horizon

_QUANTILES = [0.1, 0.5, 0.9]

# Module-level cache so the (heavy) pipeline loads once per process, keyed by model id.
# Guarded by a lock so a concurrent first-load is safe even when a caller invokes predict()
# directly (the experiment runner also serializes via its global _SETUP_LOCK; this is belt-and-
# suspenders and matches the TimesFM path).
_PIPELINE_CACHE: dict[str, object] = {}
_PIPELINE_LOCK = threading.Lock()


def _load_pipeline(model_id: str):
    with _PIPELINE_LOCK:
        if model_id in _PIPELINE_CACHE:
            return _PIPELINE_CACHE[model_id]
        import torch
        from chronos import BaseChronosPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        pipe = BaseChronosPipeline.from_pretrained(model_id, device_map=device, torch_dtype=dtype)
        _PIPELINE_CACHE[model_id] = pipe
    return pipe


@FORECASTERS.register("chronos2")
class Chronos2Forecaster:
    """Zero-shot quantile forecasts from ``amazon/chronos-2``."""

    name = "chronos2"

    def __init__(self, model_id: str = "amazon/chronos-2"):
        self.model_id = model_id

    def predict(self, item_id: str, history: pd.Series, target_weeks: Sequence[int]) -> list[BaseForecast]:
        import torch

        h = horizon(history, target_weeks)
        pipe = _load_pipeline(self.model_id)
        context = torch.tensor(history.to_numpy(dtype="float32"))
        # Chronos-2: returns (quantiles, mean) as lists (one per input series);
        # each quantiles tensor has shape (n_variates, prediction_length, n_levels).
        quantiles, _mean = pipe.predict_quantiles(
            inputs=[context], prediction_length=h, quantile_levels=_QUANTILES
        )
        q = quantiles[0].float().cpu().numpy()
        if q.ndim == 3:                          # (n_variates, h, n_levels) -> (h, n_levels)
            q = q[0]
        last = int(history.index[-1])
        by_pos: dict[int, BaseForecast] = {}
        for step in range(h):
            lo, mid, hi = sorted(float(v) for v in q[step])
            by_pos[last + 1 + step] = BaseForecast(
                point=clamp_nonneg(mid), lower=clamp_nonneg(lo), upper=clamp_nonneg(hi)
            )
        return [by_pos[int(w)] for w in target_weeks]


@FORECASTERS.register("chronos_bolt")
class ChronosBoltForecaster(Chronos2Forecaster):
    """Zero-shot quantile forecasts from ``amazon/chronos-bolt-base`` — the predecessor
    Chronos-family model (a T5-based patch quantile regressor; faster, different accuracy/bias
    profile than Chronos-2). Same ``BaseChronosPipeline.predict_quantiles`` API (arXiv 2510.15821
    lists Chronos-Bolt as a baseline)."""

    name = "chronos_bolt"

    def __init__(self, model_id: str = "amazon/chronos-bolt-base"):
        self.model_id = model_id


# --- TimesFM 2.5 (Google) — Chronos-2 baseline (arXiv 2510.15821) --------------------
_TIMESFM_CACHE: dict = {}
_TIMESFM_LOCK = threading.Lock()


def _load_timesfm(max_context: int, max_horizon: int):
    with _TIMESFM_LOCK:
        if "m" not in _TIMESFM_CACHE:
            import timesfm
            m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
            m.compile(timesfm.ForecastConfig(
                max_context=max_context, max_horizon=max_horizon, normalize_inputs=True,
                use_continuous_quantile_head=True, infer_is_positive=True))
            _TIMESFM_CACHE["m"] = m
        return _TIMESFM_CACHE["m"]


@FORECASTERS.register("timesfm25")
class TimesFM25Forecaster:
    """Zero-shot forecasts from Google TimesFM-2.5 (200M, torch). Quantile head returns deciles;
    index 1 = P10, 5 = P50, 9 = P90 (index 0 = mean). A Chronos-2 baseline."""

    name = "timesfm25"

    def __init__(self, max_context: int = 512, max_horizon: int = 64):
        self.max_context = max_context
        self.max_horizon = max_horizon

    def predict(self, item_id: str, history: pd.Series, target_weeks: Sequence[int]) -> list[BaseForecast]:
        m = _load_timesfm(self.max_context, self.max_horizon)
        h = horizon(history, target_weeks)
        y = history.to_numpy(dtype="float32")
        with _TIMESFM_LOCK:
            pt, q = m.forecast(horizon=h, inputs=[y])
        pt = np.asarray(pt); q = np.asarray(q)          # pt (1,h); q (1,h,10) deciles
        last = int(history.index[-1])
        by_pos = {last + 1 + s: BaseForecast(
            point=clamp_nonneg(float(pt[0, s])),
            lower=clamp_nonneg(float(q[0, s, 1])), upper=clamp_nonneg(float(q[0, s, 9])))
            for s in range(h)}
        return [by_pos[int(w)] for w in target_weeks]
