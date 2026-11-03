"""
M5 Walmart data source for Vendster.

Downloads the M5 dataset from a public mirror (no Kaggle credentials), aggregates
daily unit sales to weekly series, and defines the **environment pool**:
stable-price items (sell_price CV < 5%) that sell at least ``min_weekly`` units in
every week (a consistently-demanded, mature product). Negotiation baskets of 5 are sampled from this pool per episode/set, with
a fixed 2/2/1 unit-cost mix (compatible/integrative/distributive) and a single
literature-grounded elasticity (Bijmolt 2005 overall mean, -2.6) applied to every
product (product type is not a study variable). The same pool is the evaluation
population for the forecaster census.

Pure numeric helpers are separated from I/O for unit testing.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.registry import DATA_SOURCES
from ..core.types import Dataset, ProductSpec, TemporalWindows

# Public mirror of the full M5 dataset (calendar, sell_prices, sales files).
M5_ZIP_URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"

# Single literature-grounded promotional price elasticity for ALL products: the Bijmolt et al.
# (2005) meta-analysis overall mean (~ -2.62, rounded to -2.6). Product type / category is NOT a
# study variable — every product shares this elasticity, so forecasters compete only on
# base-demand accuracy, and the constant-elasticity DGP stays clean. (The per-category
# split was dropped: the stable-price pool is ~all FOODS, so category variation was negligible.)
OVERALL_ELASTICITY = -2.6

MIN_WEEKLY = 5              # pool floor: sold >= this many units in EVERY week
# Fixed negotiation-dynamics unit-cost mix for a 5-basket:
# 2 compatible (~0.20 cost/price), 2 integrative (~0.45), 1 distributive (~0.70).
COST_MIX: tuple[float, ...] = (0.20, 0.20, 0.45, 0.45, 0.70)


# --- Pure helpers -------------------------------------------------------------

def assign_elasticity(category: str, override: float | None = None) -> float:
    """Constant-elasticity coefficient. Single overall Bijmolt mean (-2.6) for every product
    (category ignored), unless an explicit override is given."""
    if override is not None:
        return float(override)
    return OVERALL_ELASTICITY


def assign_cost_mix(rng: np.random.Generator) -> list[float]:
    """A shuffled copy of the 2/2/1 unit-cost mix, assigned across a basket's 5 products."""
    mix = list(COST_MIX)
    rng.shuffle(mix)
    return mix


def price_cv(prices: pd.DataFrame) -> pd.Series:
    """Coefficient of variation of ``sell_price`` per (store_id, item_id)."""
    g = prices.groupby(["store_id", "item_id"])["sell_price"]
    stats = g.agg(["mean", "std"])
    return (stats["std"].fillna(0.0) / stats["mean"]).rename("price_cv")


def stable_price_items(prices: pd.DataFrame, threshold: float = 0.05) -> set:
    """Return {(store_id, item_id)} whose sell_price CV is below ``threshold``."""
    cv = price_cv(prices)
    return set(cv.index[cv < threshold])


def series_stats(weekly: pd.DataFrame) -> pd.DataFrame:
    """Per-series mean, cv, and min weekly units (indexed like ``weekly``)."""
    v = weekly.to_numpy(dtype=float)
    mean = v.mean(axis=1)
    cv = np.divide(v.std(axis=1), mean, out=np.zeros_like(mean), where=mean > 0)
    return pd.DataFrame({"mean": mean, "cv": cv, "min": v.min(axis=1)}, index=weekly.index)


def temporal_windows(n_weeks: int, eval_len: int = 16, event_gap: int = 4) -> TemporalWindows:
    """Standard windows: months 1-4 at the tail, a 16-week eval window before them."""
    m4 = n_weeks - 1
    events = [m4 - 3 * event_gap, m4 - 2 * event_gap, m4 - event_gap, m4]
    eval_start = events[0] - eval_len
    if eval_start < 1:
        raise ValueError(f"Not enough weeks ({n_weeks}) for windows")
    w = TemporalWindows(train_end=eval_start,
                        eval_weeks=list(range(eval_start, events[0])), event_weeks=events)
    w.validate()
    return w


# --- DataSource ---------------------------------------------------------------

@DATA_SOURCES.register("m5")
class M5DataSource:
    """Environment pool (stable-price + min-week floor) with sampled 5-baskets."""

    name = "m5"

    def __init__(self, cache_dir: str = "data/m5_cache", price_cv_threshold: float = 0.05,
                 min_weekly: int = MIN_WEEKLY, panel_size: int = 2000, eval_weeks: int = 16,
                 seed: int = 0):
        self.cache_dir = Path(cache_dir)
        self.price_cv_threshold = price_cv_threshold
        self.min_weekly = min_weekly
        self.panel_size = panel_size
        self.eval_weeks = eval_weeks
        self.seed = seed
        self._loaded = False

    # -- IO --
    def _ensure_download(self) -> None:
        if (self.cache_dir / "sales_train_evaluation.csv").exists():
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(M5_ZIP_URL, headers={"User-Agent": "vendster"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(self.cache_dir)

    def _read_csv(self, name: str) -> pd.DataFrame:
        for path in (self.cache_dir / name, *self.cache_dir.rglob(name)):
            if path.exists():
                return pd.read_csv(path)
        raise FileNotFoundError(f"{name} not found under {self.cache_dir}")

    @staticmethod
    def _weekly_matrix(sales: pd.DataFrame, calendar: pd.DataFrame):
        """Weekly units [series x wm_yr_wk]; day d_i -> i-th calendar row's wm_yr_wk."""
        d_cols = [c for c in sales.columns if c.startswith("d_")]
        wk_by_day = calendar["wm_yr_wk"].to_numpy()
        wk_of = pd.Series({c: wk_by_day[int(c.split("_")[1]) - 1] for c in d_cols})
        daily = sales[d_cols].copy()
        daily.index = pd.MultiIndex.from_tuples(list(zip(sales["store_id"], sales["item_id"])),
                                                names=["store_id", "item_id"])
        weekly = daily.T.groupby(wk_of).sum().T.reindex(sorted(set(wk_of)), axis=1)
        return weekly

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._ensure_download()
        self._calendar = self._read_csv("calendar.csv")
        self._prices = self._read_csv("sell_prices.csv")
        self._sales = self._read_csv("sales_train_evaluation.csv")
        self._weekly = self._weekly_matrix(self._sales, self._calendar)
        self._stats = series_stats(self._weekly)
        stable = stable_price_items(self._prices, self.price_cv_threshold)
        s = self._stats
        self._pool = sorted(k for k in s.index
                            if k in stable and s.loc[k, "min"] >= self.min_weekly)
        # broad panel for global-forecaster training (stable + active mean>=1)
        self._panel = sorted(k for k in s.index if k in stable and s.loc[k, "mean"] >= 1.0)
        self._med_price = self._prices.groupby(["store_id", "item_id"])["sell_price"].median()
        self._cat_of = dict(zip(zip(self._sales["store_id"], self._sales["item_id"]),
                                self._sales["cat_id"]))
        self._dept_of = dict(zip(zip(self._sales["store_id"], self._sales["item_id"]),
                                 self._sales["dept_id"]))
        self._windows = temporal_windows(self._weekly.shape[1], eval_len=self.eval_weeks)
        self._loaded = True

    # -- pool / basket --
    def pool(self) -> list[tuple]:
        self._ensure_loaded()
        return list(self._pool)

    def sample_basket(self, rng: np.random.Generator, size: int = 5) -> list[tuple]:
        self._ensure_loaded()
        if len(self._pool) < size:
            raise ValueError(f"pool has {len(self._pool)} < {size} products")
        idx = rng.choice(len(self._pool), size=size, replace=False)
        return [self._pool[i] for i in idx]

    def _uid(self, key: tuple) -> str:
        store_id, item_id = key
        return f"{item_id}_{store_id}"

    def _series(self, key: tuple) -> pd.Series:
        s = self._weekly.loc[key].reset_index(drop=True)
        s.name = self._uid(key)
        return s

    def build_dataset(self, basket_keys: Sequence[tuple] | None = None,
                      cost_seed: int = 0) -> Dataset:
        """Build a Dataset for a 5-basket (sampled if ``basket_keys`` is None)."""
        self._ensure_loaded()
        if basket_keys is None:
            basket_keys = self.sample_basket(np.random.default_rng(self.seed))
        cost_rng = np.random.default_rng(cost_seed)
        ratios = assign_cost_mix(cost_rng) if len(basket_keys) == 5 else [0.45] * len(basket_keys)

        products, weekly_out = [], {}
        for key, ucr in zip(basket_keys, ratios):
            uid = self._uid(key)
            row = self._stats.loc[key]
            weekly_out[uid] = self._series(key)
            products.append(ProductSpec(
                item_id=uid, department=str(self._dept_of[key]), category=str(self._cat_of[key]),
                price=float(self._med_price.loc[key]),
                base_demand_mean=float(row["mean"]), base_demand_cv=float(row["cv"]),
                elasticity=assign_elasticity(str(self._cat_of[key])), unit_cost_ratio=ucr,
            ))

        panel_keys = list(dict.fromkeys(list(basket_keys) + self._panel[: self.panel_size]))
        train_panel = {self._uid(k): self._series(k) for k in panel_keys}
        exog = self._build_exog(self._calendar, self._prices, self._sales,
                                list(self._weekly.columns), panel_keys)
        return Dataset(products=products, weekly=weekly_out, windows=self._windows,
                       meta={"source": "m5", "n_weeks": self._weekly.shape[1],
                             "pool_size": len(self._pool), "basket": [self._uid(k) for k in basket_keys]},
                       train_panel=train_panel, exog=exog)

    def load(self) -> Dataset:
        """Default single-instance dataset: one sampled basket (seed)."""
        return self.build_dataset()

    @staticmethod
    def _build_exog(calendar, prices, sales, weekly_cols, series_keys) -> dict:
        """Per-series weekly known-covariate frames aligned to ``weekly_cols``."""
        cal = calendar.groupby("wm_yr_wk")
        month = cal["month"].first().reindex(weekly_cols).fillna(0).to_numpy()
        event = (cal["event_name_1"].apply(lambda s: int(s.notna().any()))
                 .reindex(weekly_cols).fillna(0).to_numpy())
        woy = np.array([int(wk) % 100 for wk in weekly_cols], dtype=float)
        snap = {st: cal[f"snap_{st}"].mean().reindex(weekly_cols).fillna(0).to_numpy()
                for st in ["CA", "TX", "WI"]}
        pv = prices.pivot_table(index=["store_id", "item_id"], columns="wm_yr_wk",
                                values="sell_price").reindex(columns=weekly_cols)
        static = sales.set_index(["store_id", "item_id"])[["dept_id", "cat_id", "state_id"]]
        dept_c = {d: i for i, d in enumerate(sorted(sales["dept_id"].unique()))}
        cat_c = {c: i for i, c in enumerate(sorted(sales["cat_id"].unique()))}
        state_c = {s: i for i, s in enumerate(sorted(sales["state_id"].unique()))}
        exog = {}
        for key in series_keys:
            store_id, item_id = key
            uid = f"{item_id}_{store_id}"
            st = static.loc[key, "state_id"]
            price_arr = pv.loc[key].to_numpy() if key in pv.index else np.full(len(weekly_cols), np.nan)
            price = pd.Series(price_arr).ffill().bfill().fillna(0.0).to_numpy()
            exog[uid] = pd.DataFrame({
                "log_price": np.log1p(price),
                "snap": snap[st] if st in snap else np.zeros(len(weekly_cols)),
                "event": event, "month": month, "woy": woy,
                "dept": float(dept_c[static.loc[key, "dept_id"]]),
                "cat": float(cat_c[static.loc[key, "cat_id"]]),
                "state": float(state_c[st]),
            })
        return exog
