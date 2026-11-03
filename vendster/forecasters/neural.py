"""
Global deep-learning forecasters via ``neuralforecast`` (Nixtla): NHITS, PatchTST, DeepAR (the
study roster; DLinear remains selectable but is not in the reference roster).

These are Chronos / Chronos-2 baselines (arXiv 2403.07815 / 2510.15821) added for forecaster
DIVERSITY — they spread far more on accuracy and (signed) bias than the near-unbiased
classical/foundation set, which is what the ecological quality/bias -> outcome gradient needs.

Trained GLOBALLY on the broad panel with a leakage-safe cutoff (only pre-cutoff weeks), then
applied 1-step-ahead (the negotiation's month-4 event immediately follows the truncated history).
One fit per (model, cutoff), cached at module level so it is reused across all baskets/episodes
that share a cutoff (a per-cell cache would refit ~50x). Intervals from MQLoss 80% quantiles
(point models) or DistributionLoss quantiles (DeepAR).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

import numpy as np
import pandas as pd

# neuralforecast/pytorch-lightning are chatty (per-epoch progress) — quiet them for clean,
# reproducible benchmark logs (does not affect results).
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

from ..core.registry import FORECASTERS
from ..core.types import BaseForecast
from .base import clamp_nonneg

_MIN_LEN = 60            # skip series shorter than this in the training panel
_INPUT = 104            # context length (2 seasons); clipped to series length
_WINDOW_MARGIN = 24     # leave >= this many steps past input_size so every model (esp. DeepAR's
                        # autoregressive windowing) can form training windows on the shortest cutoffs
_LEVEL = 80             # 80% interval -> lo-80 = P10, hi-80 = P90

# module-level fit cache: (model, cutoff, input_size, max_steps) -> fitted NeuralForecast
_NF_CACHE: dict = {}
_NF_LOCK = threading.Lock()


def _make_model(model: str, h: int, input_size: int, max_steps: int):
    from neuralforecast.losses.pytorch import DistributionLoss, MQLoss
    from neuralforecast.models import NHITS, DeepAR, DLinear, PatchTST

    common = dict(h=h, input_size=input_size, max_steps=max_steps, scaler_type="standard",
                  enable_progress_bar=False, logger=False, enable_model_summary=False)
    if model == "dlinear":
        return DLinear(loss=MQLoss(level=[_LEVEL]), **common)
    if model == "nhits":
        return NHITS(loss=MQLoss(level=[_LEVEL]), **common)
    if model == "patchtst":
        return PatchTST(loss=MQLoss(level=[_LEVEL]), **common)
    if model == "deepar":
        # DeepAR is probabilistic (DistributionLoss); quantiles are requested at predict time.
        return DeepAR(loss=DistributionLoss(distribution="Normal", level=[_LEVEL]), **common)
    raise ValueError(f"unknown neural model: {model}")


@FORECASTERS.register("neural_global")
class GlobalNeuralForecaster:
    """A neuralforecast deep model trained global on the panel (leakage-safe cutoff), applied
    1-step-ahead. ``model`` in {dlinear, nhits, patchtst, deepar}."""

    name = "neural_global"
    REQUIRES = ("panel", "exog", "train_cutoff")   # capability: harness injects panel-training context

    def __init__(self, panel: dict | None = None, exog: dict | None = None,
                 train_cutoff: int | None = None, model: str = "nhits",
                 max_steps: int = 300):
        self.panel = panel or {}
        self.exog = exog or {}
        self.train_cutoff = train_cutoff
        self.model = model
        self.max_steps = max_steps
        self.input_size = _INPUT
        self._nf = None

    def _panel_long(self) -> pd.DataFrame:
        cutoff = self.train_cutoff
        frames = []
        for uid, s in self.panel.items():
            a = s.to_numpy(dtype=float)
            a = a[:cutoff] if cutoff else a
            if len(a) < _MIN_LEN:
                continue
            frames.append(pd.DataFrame({"unique_id": uid, "ds": np.arange(len(a)), "y": a}))
        if not frames:
            raise ValueError("empty training panel for global neural forecaster")
        return pd.concat(frames, ignore_index=True)

    def _fit(self):
        from neuralforecast import NeuralForecast
        key = (self.model, self.train_cutoff, self.input_size, self.max_steps)
        with _NF_LOCK:
            if key in _NF_CACHE:
                self._nf, self.input_size = _NF_CACHE[key]      # restore fitted input_size on hit
                return
            df = self._panel_long()
            min_len = int(df.groupby("unique_id").size().min())
            self.input_size = max(16, min(_INPUT, min_len - _WINDOW_MARGIN))
            nf = NeuralForecast(models=[_make_model(self.model, 1, self.input_size, self.max_steps)],
                                freq=1)
            nf.fit(df=df, verbose=False)
            _NF_CACHE[key] = (nf, self.input_size)
            self._nf = nf

    def predict(self, item_id: str, history: pd.Series, target_weeks: Sequence[int]) -> list[BaseForecast]:
        if self._nf is None:
            self._fit()
        a = history.to_numpy(dtype=float)
        # NF forecasts h=1 after the provided series end; feed the (leakage-safe) truncated history.
        ctx = a[-self.input_size:] if len(a) > self.input_size else a
        df = pd.DataFrame({"unique_id": item_id, "ds": np.arange(len(ctx)), "y": ctx})
        with _NF_LOCK:                                   # torch model fwd is not thread-safe
            fc = self._nf.predict(df=df).reset_index()
        cols = list(fc.columns)
        base = next(c for c in cols if c.endswith("-median")) if any(c.endswith("-median") for c in cols) \
            else next(c for c in cols if c not in ("unique_id", "ds") and "-lo-" not in c and "-hi-" not in c)
        lo = next((c for c in cols if c.endswith(f"-lo-{_LEVEL}")), base)
        hi = next((c for c in cols if c.endswith(f"-hi-{_LEVEL}")), base)
        row = fc.iloc[0]
        bf = BaseForecast(point=clamp_nonneg(float(row[base])),
                          lower=clamp_nonneg(float(row[lo])), upper=clamp_nonneg(float(row[hi])))
        return [bf for _ in target_weeks]
