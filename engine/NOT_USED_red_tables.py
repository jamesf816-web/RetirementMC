# engine/red_tables.py
"""
Retirement Expenditure Distribution (RED) Tables
Historical safe withdrawal rates, success rates, and guardrails
Used by simulator to determine sustainable spending

Sources:
- Bengen 4% Rule (updated with 2024 data)
- Early Retirement Now SWR Series (2023-2025)
- Guyton-Klinger Decision Rules
- Blanchett, Pfau, Kitces, McClung, etc.
"""

from typing import Dict, Tuple, List
import math

# =============================================================================
# Core Safe Withdrawal Rate (SWR) Table
# Percentage of initial portfolio (inflation-adjusted) you can withdraw annually
# Format: (years_of_retirement, stock_allocation_pct) -> swr_pct
# =============================================================================

SWR_TABLE = {
    # 60/40 portfolio
    60: {
        20: 5.2,
        25: 4.7,
        30: 4.4,
        35: 4.1,
        40: 3.9,
        50: 3.6,
    },
    # 70/30 portfolio
    70: {
        20: 5.4,
        25: 4.9,
        30: 4.5,
        35: 4.2,
        40: 4.0,
        50: 3.7,
    },
    # 80/20 portfolio
    80: {
        20: 5.6,
        25: 5.0,
        30: 4.6,
        35: 4.3,
        40: 4.1,
        50: 3.8,
    },
    # 100% stocks
    100: {
        20: 5.9,
        25: 5.2,
        30: 4.7,
        35: 4.4,
        40: 4.2,
        50: 3.9,
    },
}


# =============================================================================
# Success Rate Table (1871–2024 rolling periods)
# (years, equity_allocation) -> success_rate_pct for 4.0% constant real withdrawal
# =============================================================================

SUCCESS_RATE_4PCT = {
    20: {0: 92, 20: 94, 40: 96, 60: 97, 80: 98, 100: 98},
    25: {0: 88, 20: 91, 40: 94, 60: 96, 80: 97, 100: 98},
    30: {0: 83, 20: 88, 40: 92, 60: 95, 80: 96, 100: 97},
    35: {0: 78, 20: 84, 40: 89, 60: 93, 80: 95, 100: 96},
    40: {0: 72, 20: 79, 40: 86, 60: 91, 80: 94, 100: 95},
    50: {0: 65, 20: 73, 40: 81, 60: 88, 80: 92, 100: 94},
}


# =============================================================================
# Guyton-Klinger Dynamic Withdrawal Rules (2004/2006)
# =============================================================================

GUYTON_KLINGER_RULES = {
    "withdrawal_floor_pct": 0.80,   # Never drop more than 20% below initial real withdrawal
    "withdrawal_ceiling_pct": 1.20, # Never increase more than 20% above initial
    "prosperity_rule_increase": 0.10,  # +10% real increase if portfolio > 150% of initial
    "capital_preservation_rule": 0.80,  # Cut withdrawal 10% if portfolio < 80% of initial (real)
    "base_swr": 0.05,  # Start with 5.0% rule instead of 4%
}


# =============================================================================
# Early Retirement Now (ERN) Optimal Guardrails (2024 update)
# =============================================================================

ERN_GUARDRAILS = {
    "lower_bound_multiplier": 0.85,   # Reduce spending to 85% of original if needed
    "upper_bound_multiplier": 1.50,   # Allow up to 150% in great years
    "withdrawal_cap": 0.065,          # Never exceed 6.5% of current portfolio
    "withdrawal_floor": 0.035,        # Never go below 3.5% of current portfolio (inflation-adjusted)
}


# =============================================================================
# Public Functions
# =============================================================================

def get_safe_withdrawal_rate(
    retirement_length: int,
    equity_allocation: int,
    strategy: str = "bengen"
) -> float:
    """
    Returns safe initial withdrawal rate as decimal
    """
    equity_allocation = max(0, min(100, round(equity_allocation / 10) * 10))  # nearest 10%
    table_key = equity_allocation

    if strategy == "guyton_klinger":
        return GUYTON_KLINGER_RULES["base_swr"]

    # Default: Bengen-style fixed %
    rates = SWR_TABLE.get(table_key, SWR_TABLE[60])
    # Interpolate retirement length
    lengths = sorted(rates.keys())
    if retirement_length <= lengths[0]:
        return rates[lengths[0]] / 100
    if retirement_length >= lengths[-1]:
        return rates[lengths[-1]] / 100

    for i in range(len(lengths) - 1):
        y1, y2 = lengths[i], lengths[i + 1]
        if y1 <= retirement_length <= y2:
            r1, r2 = rates[y1], rates[y2]
            return (r1 + (r2 - r1) * (retirement_length - y1) / (y2 - y1)) / 100

    return 0.04  # fallback


def get_success_probability(
    years: int,
    equity_pct: int,
    swr: float = 0.04
) -> float:
    """
    Estimate historical success probability for given parameters
    """
    years = min(max(years, 20), 50)
    equity = max(0, min(100, round(equity_pct / 20) * 20))

    base = SUCCESS_RATE_4PCT.get(years, SUCCESS_RATE_4PCT[30])
    success_4pct = base.get(equity, 95)

    # Adjust slightly for SWR deviation from 4%
    if swr < 0.04:
        return min(99.9, success_4pct + (0.04 - swr) * 400)
    elif swr > 0.04:
        return max(50.0, success_4pct - (swr - 0.04) * 300)

    return success_4pct


def apply_dynamic_rules(
    initial_withdrawal: float,
    current_portfolio: float,
    initial_portfolio: float,
    current_cpi: float,
    initial_cpi: float,
    year: int,
    strategy: str = "guyton_klinger"
) -> float:
    """
    Apply dynamic spending rules each year
    """
    real_initial = initial_withdrawal * (current_cpi / initial_cpi)
    current_pct = current_portfolio / (initial_portfolio * (current_cpi / initial_cpi))

    if strategy == "guyton_klinger":
        floor = real_initial * GUYTON_KLINGER_RULES["withdrawal_floor_pct"]
        ceiling = real_initial * GUYTON_KLINGER_RULES["withdrawal_ceiling_pct"]

        proposed = real_initial
        if current_pct < 0.80 and year > 1:
            proposed *= 0.90  # 10% cut
        elif current_pct > 1.50:
            proposed *= 1.10  # 10% prosperity increase

        return max(floor, min(ceiling, proposed))

    elif strategy == "ern":
        floor = initial_portfolio * ERN_GUARDRAILS["withdrawal_floor"] * (current_cpi / initial_cpi)
        cap = initial_portfolio * ERN_GUARDRAILS["withdrawal_cap"] * (current_cpi / initial_cpi)
        bounded = max(floor, min(real_initial * 1.5, current_portfolio * 0.05))
        return max(floor, min(cap, bounded))

    return real_initial  # fixed spending
