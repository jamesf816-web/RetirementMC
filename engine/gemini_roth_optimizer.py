import numpy as np
import pandas as pd
from config.user_input import *
from engine import tax_utils
from engine.tax_utils import get_current_brackets

def optimal_roth_conversion(
    year,
    traditional_balance,
    taxable_income_before_conv,
    max_conversion=None,
    inflation_index=None,
    fill_target=500000
):
    """
    Returns optimal Roth conversion amount for a given year by parsing strategy strings
    and finding the appropriate bracket/threshold ceiling.
    """
    if traditional_balance <= 0:  
        return 0

    # Initial conversion limit is defined by fill_target passed to function
    room_in_bracket = fill_target - taxable_income_before_conv
    conversion = min(traditional_balance, max(0, room_in_bracket))

    # --- 3. Apply External Limits ---
    
    # Apply max conversion cap if provided
    if max_conversion is not None:
        conversion = min(conversion, max_conversion)

    # Respect withdrawal ceiling if defined
    if "withdrawal_ceiling" in globals():
        conversion = min(conversion, max(0, withdrawal_ceiling - taxable_income_before_conv))

    # --- 4. Final Return ---
    if conversion < 1000:
        return max(0, conversion)
    else:
        # nearest $1,000
        return max(0, round(conversion, -3))
    
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


