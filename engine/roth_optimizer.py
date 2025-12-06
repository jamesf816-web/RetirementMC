import numpy as np
import pandas as pd
from typing import Dict, Any

# We only need to import the dedicated tax planning function for targets.
# The internal logic of the tax planning function will use the correct tax_utils.
from engine.tax_planning import get_tax_planning_targets 

def optimal_roth_conversion(
    year: int, 
    inflation_index: float, 
    filing_status: str, 
    AGI_base: float, # Taxable income *before* conversion (RMDs, SS, Pension, etc.)
    accounts: Dict[str, Any], 
    tax_strategy: str, 
    irmaa_strategy: str
) -> float:
    """
    Calculates the optimal Roth conversion amount to fill a strategic tax bracket 
    ceiling or IRMAA threshold for a given year, based on the current AGI_base.
    """
    
    # 1. Get Traditional Balance
    traditional_balance = accounts.get("Traditional", {}).get("balance", 0.0) 

    if traditional_balance <= 0:
        return 0.0

    # 2. Get the target AGI and MAGI thresholds using the dedicated function
    # Note: get_tax_planning_targets handles fetching constants and applying inflation/status.
    tax_target_AGI, irmaa_target_MAGI = get_tax_planning_targets(
        year=year,
        inflation_this_year=inflation_index, 
        tax_strategy=tax_strategy,
        irmaa_strategy=irmaa_strategy,
        filing_status=filing_status
    )
    
    # Initialization
    conversion = 0.0 
    
    # 3. Calculate conversion room based on Tax Strategy
    # The room is the difference between the ceiling and the current AGI_base.
    room_in_tax_bracket = tax_target_AGI - AGI_base
    
    # Initial conversion amount is capped by the room available in the tax bracket
    # and the remaining traditional balance.
    conversion = min(traditional_balance, max(0.0, room_in_tax_bracket))

    # 4. Apply IRMAA Threshold (if lower)
    # IRMAA thresholds also act as a ceiling to the total MAGI (which conversion contributes to).
    room_in_irmaa_threshold = irmaa_target_MAGI - AGI_base
    
    # The conversion amount is capped again by the room available under the IRMAA threshold.
    conversion = min(conversion, max(0.0, room_in_irmaa_threshold))

    # 5. Final Rounding/Threshold Check
    # Removed unsupported max_conversion/withdrawal_ceiling logic.
    
    if conversion < 1000:
        return max(0.0, conversion)
    else:
        # Round to nearest $1,000 for practicality
        return max(0.0, round(conversion, -3)) 

# --- Corrected Placeholder for get_conversion_plan (Not used by main simulator) ---
def get_conversion_plan(
    start_year: int = 2026, 
    end_year: int = 2035, 
    filing_status: str = "married_filing_jointly",
    traditional_balance: float = 500000, 
    base_income_per_year: Dict[int, float] = None, 
    strategy: str = "fill_24_percent", 
    tier: str = "fill_IRMAA_3"
) -> pd.DataFrame:
    """
    Returns a DataFrame with recommended Roth conversion each year (simplified batch run)
    """
    
    # Placeholder for inflation index since it's not simulated here
    INFLATION_INDEX_PLACEHOLDER = 1.0 

    if base_income_per_year is None:
        base_income_per_year = {y: 100_000.0 for y in range(start_year, end_year + 1)}

    plan = []
    current_traditional_balance = traditional_balance # Use a mutable variable
    
    for y in range(start_year, end_year + 1):
        conv = optimal_roth_conversion(
            year=y,
            inflation_index=INFLATION_INDEX_PLACEHOLDER, 
            filing_status=filing_status,
            AGI_base=base_income_per_year.get(y, 0.0), 
            accounts={"Traditional": {"balance": current_traditional_balance}}, # Pass current balance
            tax_strategy=strategy,
            irmaa_strategy=tier,
        )
        current_traditional_balance = max(0.0, current_traditional_balance - conv) # Adjust balance
        
        plan.append({
            "year": y, 
            "conversion": conv, 
            "strategy": strategy, 
            "IRMAA": tier, 
            "remaining_trad_balance": current_traditional_balance
        })

    return pd.DataFrame(plan)
