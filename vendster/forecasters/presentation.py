"""
Turn base-demand forecasts into the Buyer-facing per-discount forecast tables.

The identical constant-elasticity factor ``(1 - d)^(-|beta|)`` (per product) is applied to
every forecaster's base prediction (see :meth:`BaseForecast.scaled`), so forecasters compete
only on base-demand accuracy. ``|beta|`` is per-product (``ProductSpec.elasticity``), so a
bring-your-own data source can vary elasticity across products freely.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.types import BaseForecast, Dataset


def scaled_table(base: BaseForecast, abs_beta: float, discount_levels: Sequence[float]) -> dict[float, BaseForecast]:
    """Per-discount forecast for one product from its base-demand forecast."""
    return {float(d): base.scaled(abs_beta, float(d)) for d in discount_levels}


def build_event_forecast(
    forecaster,
    dataset: Dataset,
    event_week: int,
    discount_levels: Sequence[float],
) -> dict[str, dict[float, BaseForecast]]:
    """Forecast month-4 base demand per product (history strictly before the event),
    scaled to each discount level. ``forecaster`` may be a perturbation forecaster
    (which reads the true series) or a real one (which uses ``history``).
    """
    out: dict[str, dict[float, BaseForecast]] = {}
    for p in dataset.products:
        history = dataset.weekly[p.item_id].iloc[:event_week]
        base = forecaster.predict(p.item_id, history, [event_week])[0]
        out[p.item_id] = scaled_table(base, abs(p.elasticity), discount_levels)
    return out


def build_track_record(
    forecaster,
    dataset: Dataset,
    past_events: list[dict],
) -> list[dict]:
    """Forecaster predicted-vs-actual on the months 1-3 events (Buyer track record).

    For each past event, forecast base demand with history strictly before that
    event week, scale to the discount that was actually used, and compare to the
    realized units. Returns one record per (event, product).
    """
    records = []
    for ev in past_events:
        wk = ev["week"]
        for p in dataset.products:
            history = dataset.weekly[p.item_id].iloc[:wk]
            base = forecaster.predict(p.item_id, history, [wk])[0]
            d = ev["products"][p.item_id]["discount"]
            pred = base.scaled(abs(p.elasticity), d)
            actual = ev["products"][p.item_id]["units"]
            records.append({
                "month": ev["month"], "item_id": p.item_id, "discount": d,
                "predicted": pred.point, "lower": pred.lower, "upper": pred.upper,
                "actual": actual,
            })
    return records
