# tax_planning.py
#
# Used for estimated taxes. Seperate tax_engine.py does final tyear end taxes correctly.
#

import copy
from typing import Tuple, Dict, Any

# CORRECTED IMPORT: We now import the new, robust function name.
from utils.tax_utils import get_indexed_federal_constants 

def estimate_taxable_gap(
    cash_needed: float, 
    accounts: Dict, 
    accounts_bal: Dict, 
    withdrawal_order: list[str],
    filing_status: str
) -> Tuple[float, float]:
    """
    Estimates the ordinary income and LTCG income resulting from a portfolio 
    withdrawal of `cash_needed`, following the withdrawal priority.
    
    This function is a 'dry run' used for estimating taxes *before* a final withdrawal
    is executed or taxes are calculated.
    
    Args:
        cash_needed: The total cash amount to withdraw from the portfolio.
        accounts: The dictionary of account definitions (includes basis, owner, etc.).
        accounts_bal: The dictionary of current account balances.
        withdrawal_order: The strategic priority list (e.g., ["taxable", "traditional", ...]).
        
    Returns:
        tuple[float, float]: (ordinary_income, ltcg_income)
    """
    ordinary_income = 0.0
    ltcg_income = 0.0
    remaining = cash_needed
    
    # Use a copy of balances so the function doesn't modify the main simulation state
    est_tax_bal = accounts_bal.copy()

    for acct_type in withdrawal_order:
       # Find accounts matching this type (must match the logic in the simulator loop)
       target_accounts = [name for name, acct in accounts.items() if acct["tax"] == acct_type]
       
       for acct_name in target_accounts:
           acct = accounts[acct_name]
           if remaining <= 0:
               break

           acct_balance = est_tax_bal.get(acct_name, 0.0)
           if acct_balance <= 0:
               continue
               
           withdraw_amt = min(acct_balance, remaining)
           
           # --- Taxable Income Characterization ---
           if acct_type == "taxable":
               # Replicate the gain calculation logic exactly from the simulator's loop
               # This requires careful state management, relying on the basis stored in `accounts`.
               original_balance = acct.get("balance", acct_balance) 
               current_gain = max(0, original_balance - acct.get("basis", 0.0))
               
               if original_balance > 0 and current_gain > 0:
                   gain_percentage = current_gain / original_balance
                   realized_gains = withdraw_amt * gain_percentage
               else:
                   realized_gains = 0.0
                   
               # Split gains between ordinary and LTCG
               ordinary_part = realized_gains * acct.get("ordinary_pct", 0.1)
               ltcg_part = realized_gains - ordinary_part
               ordinary_income += ordinary_part
               ltcg_income += ltcg_part
               
           elif acct_type in ["inherited", "traditional", "def457b"]:
               # Tax-deferred draws are 100% ordinary income
               ordinary_income += withdraw_amt
               
           # ROTH and Trust Principle draws generate no income for tax purposes

           # Update estimated balance and remaining need
           est_tax_bal[acct_name] -= withdraw_amt
           remaining -= withdraw_amt

    return ordinary_income, ltcg_income

# --------------------------------------------------

def get_tax_planning_targets(
    year: int, 
    inflation_this_year: float, 
    tax_strategy: str, 
    irmaa_strategy: str, 
    filing_status: str
) -> Tuple[float, float]:
    """
    Determines the target tax bracket ceiling and IRMAA threshold based on 
    user-defined strategies for the current year.
    """
    
    # CORRECTED CALL: Use get_indexed_federal_constants to get the combined constants dictionary
    constants = get_indexed_federal_constants(year, inflation_this_year, filing_status)

    # Extract bracket ceilings from the constants dictionary (upper bound is index 1)
    fill_targets = {
        "fill_12_percent": constants["ord_dict"]["12_percent"][1],
        "fill_22_percent": constants["ord_dict"]["22_percent"][1],
        "fill_24_percent": constants["ord_dict"]["24_percent"][1],
        "fill_32_percent": constants["ord_dict"]["32_percent"][1],
    }
    
    # Extract IRMAA thresholds (returned as a 0-indexed list of 5 tiers)
    fill_thresholds = {
        "fill_IRMAA_1": constants["irmaa_thresholds"][0], # Tier 1 (Index 0)
        "fill_IRMAA_2": constants["irmaa_thresholds"][1], # Tier 2 (Index 1)
        "fill_IRMAA_3": constants["irmaa_thresholds"][2], # Tier 3 (Index 2)
        "fill_IRMAA_4": constants["irmaa_thresholds"][3], # Tier 4 (Index 3)
        "fill_IRMAA_5": constants["irmaa_thresholds"][4], # Tier 5 (Index 4)
    }
    
    # Apply tax strategy - fill a tax bracket
    tax_target = fill_targets.get(tax_strategy, float('inf')) 

    # Apply IRMAA threshold 
    irmaa_target = fill_thresholds.get(irmaa_strategy, float('inf'))
    
    # The actual planning target is the minimum of the two constraints
    return tax_target, irmaa_target
