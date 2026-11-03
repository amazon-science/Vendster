"""
The fixed Vendster task core.

These constants are the reference negotiation task that makes results comparable
across runs and papers. A derived instance can change the protocol settings through
``BenchmarkConfig.negotiation`` and the roles, objectives, and information split through
a ``Scenario`` (see ``vendster.core.interfaces``). Changing them for the reference
instance requires a reference-version bump.
"""

from __future__ import annotations

# Reference benchmark identity. Bump when the frozen reference config changes.
REFERENCE_VERSION = "vendster-m5-v1"

# Discrete discount action space (fraction off list price). Includes 0% (no discount) so the
# Buyer can decline to discount when it (over-)believes demand is high — a real action and the
# clean over-estimation failure mode; also widens the leverage range.
DISCOUNT_LEVELS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)

# Negotiation protocol.
MAX_TURNS: int = 20
OPENING_ROLE: str = "Buyer"          # Buyer opens with the first proposal
ROLES: tuple[str, str] = ("Buyer", "Vendor")
ACTION_TYPES: tuple[str, str] = ("proposal", "decision")
DECISIONS: tuple[str, str] = ("accept", "reject")

# Agent objectives (documented, fixed).
#   Buyer:  reach a fixed total-sales THRESHOLD across all products.
#   Vendor: maximize total portfolio PROFIT = sum_i units_i * (price_i*(1-d_i) - cost_i).
BUYER_OBJECTIVE = "threshold_sales_target"
VENDOR_OBJECTIVE = "total_portfolio_profit"

# Number of past open-history events shown to both agents (months 1-3).
N_HISTORICAL_EVENTS: int = 3
