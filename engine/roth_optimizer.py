import numpy as np
import pandas as pd
from config.user_input import *
from engine import tax_utils
from engine.tax_utils import get_current_brackets

def optimal_roth_conversion(
    year,
    traditional_balance,
    taxable_income_before_conv,
    strategy="fill_24_percent",
    tier="fill_IRMAA_3",
    max_conversion=None,
    inflation_this_year=None
):
    """
    Returns optimal Roth conversion amount for a given year.
    """
    if traditional_balance <= 0: 
        return 0

    # Determine target Federal Tax Brackets and IRMAA thresholds
    brackets, brackets_dict, thresholds = get_current_brackets(year, inflation_this_year)
    # old syntaxbrackets, thresholds = get_current_brackets(year, inflation_this_year) 
    fill_targets = {
        "fill_12_percent": brackets[1][1],
        "fill_22_percent": brackets[2][1],
        "fill_24_percent": brackets[3][1],
        "fill_32_percent": brackets[4][1],
    }
    fill_thresholds = {
        "fill_IRMAA_1": thresholds[1],
        "fill_IRMAA_2": thresholds[2],
        "fill_IRMAA_3": thresholds[3],
        "fill_IRMAA_4": thresholds[4],
    }

    conversion = 0
    magi_estimate = taxable_income_before_conv

    # Apply tax strategy - fill a tax bracket
    if strategy.startswith("fill_"):
        target_bracket = fill_targets.get(strategy)
        room_in_bracket = target_bracket - magi_estimate
        conversion = min(traditional_balance, max(0, room_in_bracket))

    # Apply IRMAA threshold if there is one
    if tier.startswith("fill_"):
        threshold = fill_thresholds.get(tier)
        room_in_bracket = threshold - magi_estimate
        conversion = min(conversion, max(0, room_in_bracket))

    # Apply max conversion cap if provided
    if max_conversion is not None:
        conversion = min(conversion, max_conversion)

    # Respect withdrawal ceiling if defined
    if "withdrawal_ceiling" in globals():
        conversion = min(conversion, max(0, withdrawal_ceiling - taxable_income_before_conv))

    if conversion < 1000:
        return max(0,conversion)
    else:
        return max(0, round(conversion, -3))  # nearest $1,000

def get_conversion_plan(start_year=2026, end_year=2035, person="JEF", traditional_balance=500000, base_income_per_year=None, strategy="fill_24_percent", tier="fill_IRMAA_3", max_conv=500000):
    """
    Returns a DataFrame with recommended Roth conversion each year
    """
    if base_income_per_year is None:
        base_income_per_year = {y: 100_000 for y in range(start_year, end_year+1)}

    plan = []
    for y in range(start_year, end_year+1):
        conv = optimal_roth_conversion(
            year=y,
            traditional_balance=traditional_balance,
            taxable_income_before_conv=base_income_per_year.get(y, 0),
            strategy=strategy,
            irmaa_tier=tier,
            max_conversion=max_conv
        )
        plan.append({"year": y, "conversion": conv, "strategy": strategy, "IRMAA": tier})

    return pd.DataFrame(plan)


