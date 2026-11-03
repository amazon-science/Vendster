"""
Negotiation scenarios — the swappable *identity* of a Vendster episode.

A Scenario defines the roles, their objectives (system prompts), the asymmetric private
information each role sees, and the success criterion. The shared *arena* (``Episode``) owns the
turn loop, retries, transcript, action validation, and DGP realization — so a new scenario is
"implement + register", no arena/harness edits. ``VendsterScenario`` is the reference buyer–vendor
promotional-discount negotiation (Buyer has a sales target + sees the forecast; Vendor maximizes
portfolio profit + holds private margins).
"""

from __future__ import annotations

from ..core import task_core
from ..core.registry import SCENARIOS


@SCENARIOS.register("vendster")
class VendsterScenario:
    """Reference scenario: Buyer (threshold SALES target, sees demand forecast) vs Vendor
    (maximize TOTAL PORTFOLIO PROFIT, private unit costs/margins)."""

    name = "vendster"
    roles = ("Buyer", "Vendor")
    opening_role = task_core.OPENING_ROLE

    def next_role(self, role: str) -> str:
        return self.roles[1] if role == self.roles[0] else self.roles[0]

    def system_prompt(self, role: str, target: float | None) -> str:
        if role == "Buyer":
            if target is None:                 # calibration (target-free) variant
                return ("You are the Buyer. Objective: negotiate a FAIR deal that balances "
                        "strong sales volume against a good vendor relationship — there is no "
                        "fixed target. Only you can see the demand forecast. Respond with one JSON action.")
            return (f"You are the Buyer. Objective: reach a total SALES target of "
                    f"${target:,.0f} across all products by negotiating discounts. "
                    "Only you can see the demand forecast. Respond with one JSON action.")
        return ("You are the Vendor. Objective: maximize TOTAL PORTFOLIO PROFIT across all "
                "products. You know unit costs/margins (private). Respond with one JSON action.")

    def role_blocks(self, role: str, episode) -> list[str]:
        """Asymmetric information: Buyer sees the forecast (+ track record); Vendor sees margins."""
        blocks: list[str] = []
        if role == "Buyer" and episode.forecast_table is not None:
            blocks.append(episode._forecast_block())
            if episode.track_record is not None:
                blocks.append(episode._track_record_block())
        elif role == "Vendor":
            blocks.append(episode._margins_block())
        return blocks

    def target_met(self, total_sales: float, target: float | None) -> bool | None:
        return (total_sales >= target) if target is not None else None
