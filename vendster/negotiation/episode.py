"""
One Vendster negotiation episode.

Buyer (threshold sales target) and Vendor (total portfolio profit) alternate
proposals/decisions over discrete discounts for the 5 products of a single event
(the stretch's month-4). The Buyer alone sees the forecast (+ optional months-1-3
track record); both see the months-1-3 past events. Month-4 demand is realized
with a single DGP draw. Leakage-safe: the forecaster is trained strictly before
the stretch (train_cutoff = m1); the target week is never in training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import task_core
from ..core.types import Dataset
from ..dgp.demand import generate_past_events, realize_event
from ..dgp.stretch_pool import StretchWindow
from ..forecasters.presentation import build_event_forecast, build_track_record
from .actions import InvalidActionError, validate_action
from .scenario import VendsterScenario


@dataclass
class EpisodeResult:
    episode_id: str
    condition_id: str
    status: str                       # "agreement" | "max_turns" | "error"
    turns: int
    agreement: bool
    agreed_discounts: dict
    total_sales: float
    vendor_profit: float
    target: float | None
    target_met: bool | None
    max_sales_possible: float
    sales_efficiency: float
    first_offer_aggressiveness: float | None
    concession_rate: float | None
    decision_regret: float
    communication_log: list = field(default_factory=list)
    context: dict = field(default_factory=dict)   # what each side privately saw (constant across turns)


class Episode:
    def __init__(self, *, demand_model, llm_client, dataset: Dataset, stretch: StretchWindow,
                 condition_id: str, forecaster, target: float | None, rng: np.random.Generator,
                 discount_levels=task_core.DISCOUNT_LEVELS, max_turns: int = task_core.MAX_TURNS,
                 max_retries: int = 3, show_track_record: bool = True, episode_id: str = "ep",
                 scenario=None, vendor_llm_client=None):
        self.dm = demand_model
        self.llm = llm_client                             # client for scenario.opening_role (the Buyer sees the forecast in vendster)
        self.vendor_llm = vendor_llm_client or llm_client  # other role; defaults to same model (self-play)
        self.ds = dataset
        self.stretch = stretch
        self.condition_id = condition_id
        self.forecaster = forecaster
        self.target = target
        self.rng = rng
        self.scenario = scenario or VendsterScenario()   # negotiation identity (roles/objectives/info)
        self.discounts = tuple(discount_levels)
        self.max_turns = max_turns
        self.max_retries = max_retries
        self.show_track_record = show_track_record
        self.episode_id = episode_id
        self.products = dataset.products
        self.pids = [p.item_id for p in dataset.products]
        self.m4 = stretch.m4

        # Leakage guard: forecaster (if any) must be trained strictly before the stretch.
        # A raised error (not `assert`, which `python -O` strips) — this is the correctness-critical
        # no-leakage invariant and must hold in every run mode.
        if stretch.train_cutoff > min(stretch.event_weeks):
            raise ValueError(
                f"leakage: train_cutoff={stretch.train_cutoff} must precede all event weeks "
                f"{stretch.event_weeks}")

    # -- setup --
    def _setup(self):
        # Idempotent: the runner may call this once under a lock (to serialize non-thread-safe
        # forecaster inference — e.g. the shared Chronos GPU pipeline) before the parallel run().
        if getattr(self, "_did_setup", False):
            return
        self._did_setup = True
        weeks = self.stretch.event_weeks[:task_core.N_HISTORICAL_EVENTS]
        self.past_events = generate_past_events(
            self.dm, _stretch_dataset(self.ds, self.stretch), self.rng, self.discounts,
            n_events=len(weeks),
        )
        self.forecast_table = None
        self.track_record = None
        if self.forecaster is not None:
            self.forecast_table = build_event_forecast(self.forecaster, self.ds, self.m4, self.discounts)
            if self.show_track_record:
                self.track_record = build_track_record(self.forecaster, self.ds, self.past_events)

    # -- prompts --
    def _system(self, role: str) -> str:
        return self.scenario.system_prompt(role, self.target)

    def _state_line(self, role: str, turn: int, active) -> str:
        ap = "none" if not active else "yes"
        return (f"[STATE] role={role} turn={turn} "
                f"products={','.join(self.pids)} "
                f"discounts={','.join(str(d) for d in self.discounts)} active_proposal={ap}")

    def _turn_prompt(self, role: str, turn: int, active, log, error: str = "") -> str:
        L = [f"=== VENDSTER NEGOTIATION — Turn {turn}/{self.max_turns} — you are the {role} ===",
             self._products_block(),
             self._past_events_block(role)]
        L.extend(self.scenario.role_blocks(role, self))     # asymmetric per-role private info
        L.append(self._active_block(active, log))
        if error:
            L.append(f"\n[ERROR] {error} — return a corrected JSON action.")
        L.append(self._format_block())
        L.append(self._state_line(role, turn, active))
        return "\n".join(filter(None, L))

    def _products_block(self) -> str:
        rows = [f"  {p.item_id}: {p.category} ${p.price:.2f}" for p in self.products]
        return ("PRODUCTS:\n" + "\n".join(rows)
                + f"\nAllowed discounts: {', '.join(f'{d:.0%}' for d in self.discounts)}")

    def _past_events_block(self, role: str) -> str:
        L = ["PAST EVENTS (both agents see units & revenue):"]
        for ev in self.past_events:
            L.append(f"  Month {ev['month']}:")
            for pid, r in ev["products"].items():
                line = f"    {pid} @ {r['discount']:.0%}: {r['units']:.0f} units, ${r['sales']:.0f}"
                if role == "Vendor":
                    line += f", profit ${r['profit']:.0f}"
                L.append(line)
        return "\n".join(L)

    def _forecast_block(self) -> str:
        L = ["FORECAST (Buyer only) — predicted month-4 units per discount [80% PI]:"]
        for pid in self.pids:
            L.append(f"  {pid}:")
            for d in self.discounts:
                bf = self.forecast_table[pid][d]
                L.append(f"    {d:.0%}: {bf.point:.0f} [{bf.lower:.0f}, {bf.upper:.0f}]")
        return "\n".join(L)

    def _track_record_block(self) -> str:
        L = ["FORECAST TRACK RECORD (predicted vs actual, months 1-3):"]
        for r in self.track_record:
            L.append(f"  M{r['month']} {r['item_id']} @ {r['discount']:.0%}: "
                     f"pred {r['predicted']:.0f} [{r['lower']:.0f},{r['upper']:.0f}] vs actual {r['actual']:.0f}")
        return "\n".join(L)

    def _margins_block(self) -> str:
        L = ["YOUR COSTS/MARGINS (Vendor private):"]
        for p in self.products:
            L.append(f"  {p.item_id}: cost ${p.unit_cost:.2f} of ${p.price:.2f}")
        return "\n".join(L)

    def _active_block(self, active, log) -> str:
        if active:
            return (f"CURRENT PROPOSAL by {active['agent']}: {active['proposal']}\n"
                    "ACCEPT or REJECT it, or make a new PROPOSAL (replaces it).")
        return "NO ACTIVE PROPOSAL — make a PROPOSAL covering all products."

    def _format_block(self) -> str:
        ex = {str(p): 0.10 for p in self.pids}
        return ('RESPOND with one JSON action in a ```json block:\n'
                '  proposal: {"type":"proposal","content":"...","proposal":' + str(ex) + '}\n'
                '  decision: {"type":"decision","content":"...","decision":"accept"|"reject"}')

    # -- loop --
    def run(self) -> EpisodeResult:
        self._setup()
        log: list = []
        active = None
        role = self.scenario.opening_role
        status = "max_turns"
        for turn in range(1, self.max_turns + 1):
            action = self._agent_turn(role, turn, active, log)
            if action is None:
                status = "error"
                break
            action["agent"] = role
            action["turn"] = turn
            log.append(action)
            if action["type"] == "proposal":
                active = action
            elif action["decision"] == "accept" and active is not None:
                status = "agreement"
                break
            else:
                # A rejected proposal is off the table; without this the proposer would be shown
                # its own rejected offer next turn and could "accept" it as an agreement.
                active = None
            role = self.scenario.next_role(role)
        return self._finalize(status, log, active, turn)

    def _agent_turn(self, role, turn, active, log):
        # Buyer (opening role, sees the forecast) uses self.llm; the other role uses self.vendor_llm.
        # For self-play both are the same client, so this is a no-op there.
        llm = self.llm if role == self.scenario.opening_role else self.vendor_llm
        error = ""
        for attempt in range(self.max_retries + 1):
            messages = self._messages(role, turn, active, log, error)
            raw = llm.generate(self._system(role), messages)
            try:
                return validate_action(raw, self.pids, self.discounts)
            except InvalidActionError as e:
                error = str(e)
        return None

    def _messages(self, role, turn, active, log, error):
        msgs = []
        for m in log:
            r = "assistant" if m["agent"] == role else "user"
            c = f"[{m['agent']}] {m.get('content','')}"
            c += f" PROPOSAL {m['proposal']}" if m["type"] == "proposal" else f" DECISION {m['decision']}"
            msgs.append({"role": r, "content": c})
        msgs.append({"role": "user", "content": self._turn_prompt(role, turn, active, log, error)})
        return msgs

    def _context_snapshot(self) -> dict:
        """The private context each side saw, rendered exactly as in the prompts and constant
        across turns (system objective, shared products/past-events, Buyer-only forecast + track
        record, Vendor-only margins). Persisted so an archived transcript is fully self-contained
        and auditable without re-running the seeded DGP."""
        return {
            "buyer_system": self._system("Buyer"),
            "vendor_system": self._system("Vendor"),
            "products": self._products_block(),                 # shared
            "past_events_buyer": self._past_events_block("Buyer"),   # shared (units & revenue)
            "past_events_vendor": self._past_events_block("Vendor"), # shared + Vendor-only profit lines
            "buyer_forecast": self._forecast_block() if self.forecast_table is not None else None,
            "buyer_track_record": self._track_record_block() if self.track_record else None,
            "vendor_margins": self._margins_block(),            # Vendor private
        }

    # -- resolution & metrics --
    def _finalize(self, status, log, active, turns) -> EpisodeResult:
        agreement = status == "agreement"
        agreed = ({pid: float(active["proposal"][pid]) for pid in self.pids}
                  if agreement and active else {pid: 0.0 for pid in self.pids})
        realized = realize_event(self.dm, self.ds, agreed, self.m4, self.rng) if agreement else None
        total_sales = realized["total_sales"] if realized else 0.0
        vendor_profit = realized["total_profit"] if realized else 0.0

        max_sales = self._max_expected_sales()
        agreed_exp = self._expected_sales(agreed) if agreement else 0.0
        regret = max(0.0, max_sales - agreed_exp)

        return EpisodeResult(
            episode_id=self.episode_id, condition_id=self.condition_id, status=status,
            turns=turns, agreement=agreement, agreed_discounts=agreed,
            total_sales=total_sales, vendor_profit=vendor_profit, target=self.target,
            target_met=self.scenario.target_met(total_sales, self.target),
            max_sales_possible=max_sales,
            sales_efficiency=(total_sales / max_sales) if max_sales > 0 else 0.0,
            first_offer_aggressiveness=self._first_offer(log),
            concession_rate=self._concession_rate(log),
            decision_regret=regret,
            communication_log=log,
            context=self._context_snapshot(),
        )

    def _expected_sales(self, discounts) -> float:
        tot = 0.0
        for p in self.products:
            base = float(self.ds.weekly[p.item_id].iloc[self.m4])
            d = discounts[p.item_id]
            tot += self.dm.expected_units(p, base, d) * p.price * (1 - d)
        return tot

    def _max_expected_sales(self) -> float:
        tot = 0.0
        for p in self.products:
            base = float(self.ds.weekly[p.item_id].iloc[self.m4])
            tot += max(self.dm.expected_units(p, base, d) * p.price * (1 - d) for d in self.discounts)
        return tot

    def _first_offer(self, log):
        for m in log:
            if m["agent"] == self.scenario.opening_role and m["type"] == "proposal":
                return float(np.mean(list(m["proposal"].values())))
        return None

    def _concession_rate(self, log):
        # mean absolute change in a proposer's successive proposal means
        per = {"Buyer": [], "Vendor": []}
        for m in log:
            if m["type"] == "proposal":
                per[m["agent"]].append(float(np.mean(list(m["proposal"].values()))))
        deltas = []
        for seq in per.values():
            deltas += [abs(seq[i] - seq[i - 1]) for i in range(1, len(seq))]
        return float(np.mean(deltas)) if deltas else None


def _stretch_dataset(ds: Dataset, stretch: StretchWindow) -> Dataset:
    """A shallow Dataset view whose windows.event_weeks are the stretch's, so
    ``generate_past_events`` uses this stretch's months 1-3."""
    from ..core.types import TemporalWindows
    w = TemporalWindows(train_end=stretch.train_cutoff,
                        eval_weeks=list(range(max(0, stretch.train_cutoff - 16), stretch.train_cutoff)),
                        event_weeks=stretch.event_weeks)
    return Dataset(products=ds.products, weekly=ds.weekly, windows=w, meta=ds.meta,
                   train_panel=ds.train_panel, exog=ds.exog)
