"""
Comprehensive U.S. tax and Medicare premium calculator for retirement planning.
It contains the final tax calculation formulas, relying entirely on indexed 
constants provided by utils.tax_utils.
"""
from typing import Dict, Tuple, Literal, List, Union
import numpy as np
import logging

# Configure logging for state tax messages
logger = logging.getLogger(__name__)

# Import all necessary external components (Constants and Core Indexer)
from utils.tax_utils import (
    get_indexed_federal_constants, 
    SS_TAX_THRESHOLDS, 
    VA_TAX_BRACKETS, 
    VA_SD_MFJ, 
    VA_PE_PER_PERSON, 
    ME_TAX_BRACKETS,         
    ME_SD_2026,              
    PART_B_SURCHARGES_MONTHLY, 
    PART_D_SURCHARGES_MONTHLY,
    TaxFilingStatus # For type hints
)

# --- 1. Internal Helper Functions ---

def _federal_income_tax(
    taxable_ordinary_base: float, 
    lt_cap_gains: float, 
    qualified_dividends: float,
    federal_constants: Dict[str, Union[float, List, Dict]],
) -> float:
    """Calculates the Federal Income Tax (Ordinary + LTCG/QDiv)."""
    
    # 1. Tax on Ordinary Income (uses indexed ordinary brackets)
    ord_tax = 0.0
    ord_brackets = federal_constants["ord_list"]
    remaining_taxable = taxable_ordinary_base
    
    for low, high, rate in ord_brackets:
        if remaining_taxable <= 0:
            break
        bracket_income = min(remaining_taxable, high - low) if np.isfinite(high) else remaining_taxable
        ord_tax += bracket_income * rate
        remaining_taxable -= bracket_income

    # 2. Tax on Preferential Income (LTCG/QDiv)
    ltcg_tax = 0.0
    total_preferential = lt_cap_gains + qualified_dividends
    cg_brackets = federal_constants["cg_list"]
    taxable_income = taxable_ordinary_base + total_preferential
    current_cg_base = taxable_ordinary_base # Where preferential income starts stacking

    for low, high, rate in cg_brackets:
        # Determine the portion of preferential income that falls into this CG bracket
        bracket_start = max(low, current_cg_base)
        bracket_end = min(high, taxable_income) if np.isfinite(high) else taxable_income
        
        taxable_in_cg_bracket = max(0, bracket_end - bracket_start)
        ltcg_tax += taxable_in_cg_bracket * rate
        
    return ord_tax + ltcg_tax

def _virginia_income_tax(AGI: float, filing_status: TaxFilingStatus) -> float:
    """Calculates the total Virginia State Income Tax."""
    
    # Determine VA State Deduction (Logic)
    if filing_status == "married_joint":
        va_deduction = VA_SD_MFJ + (2 * VA_PE_PER_PERSON)
    elif filing_status == "single":
        # Assumed standard deduction for single in VA + 1 Personal Exemption
        va_deduction = 7800 + (1 * VA_PE_PER_PERSON) 
    else:
        # Default/Simplified for other statuses
        va_deduction = 7800 + (1 * VA_PE_PER_PERSON)
        
    taxable_income_va = max(0, AGI - va_deduction)
    va_tax = 0.0

    # Apply VA_TAX_BRACKETS (Non-indexed)
    for low, high, rate in VA_TAX_BRACKETS:
        if taxable_income_va <= 0:
            break
        
        bracket_income = min(taxable_income_va, high - low) if np.isfinite(high) else taxable_income_va
        va_tax += bracket_income * rate
        taxable_income_va -= bracket_income

    return va_tax

def _maine_income_tax(AGI: float, filing_status: TaxFilingStatus, inflation_index: float) -> float:
    """Calculates the total Maine State Income Tax."""
    
    # Maine Standard Deduction (Indexed, assuming ME_SD_2026 is in utils)
    # Note: Maine deductions/brackets are state-indexed differently, 
    # but for simplicity here we use the federal inflation index if state index is not available.
    BASE_YEAR = 2026
    inflation_factor = inflation_index if filing_status in ME_SD_2026 and inflation_index >= 1.0 else 1.0

    # Get Maine State Deduction (Indexed)
    me_deduction = ME_SD_2026.get(filing_status, ME_SD_2026["married_joint"]) * inflation_factor
        
    taxable_income_me = max(0, AGI - me_deduction)
    me_tax = 0.0

    # Get Maine Brackets (Indexed)
    me_brackets = ME_TAX_BRACKETS.get(filing_status, ME_TAX_BRACKETS["married_joint"])

    # Apply ME Brackets
    for low, high, rate in me_brackets:
        if taxable_income_me <= 0:
            break
        
        # Brackets must also be indexed
        inflated_low = low * inflation_factor
        inflated_high = high * inflation_factor if np.isfinite(high) else np.inf

        bracket_income = min(taxable_income_me, inflated_high - inflated_low) if np.isfinite(inflated_high) else taxable_income_me
        me_tax += bracket_income * rate
        taxable_income_me -= bracket_income

    return me_tax

def _get_irmaa_surcharge(
    magi_two_years_ago: float, 
    age1: int, 
    age2: int, 
    filing_status: TaxFilingStatus, 
    federal_constants: Dict[str, Union[float, List, Dict]]
) -> Tuple[float, float]:
    """Calculates the annual Medicare Part B and D IRMAA surcharges."""
    
    persons_covered = 0
    if age1 >= 65:
        persons_covered += 1
    if age2 is not None and age2 >= 65:
        persons_covered += 1
        
    if persons_covered == 0:
        return 0.0, 0.0

    # 1. Get IRMAA Tier (based on MAGI two years prior and indexed thresholds)
    tiers = federal_constants["irmaa_thresholds"]
    tier_index = 0
    for i, threshold in enumerate(tiers):
        if magi_two_years_ago < threshold:
            tier_index = i
            break
    else:
        tier_index = len(tiers) # Highest tier (Tier 5)

    # 2. Get Surcharge Amounts (Non-indexed constants imported from utils)
    part_b_surcharge_mo = PART_B_SURCHARGES_MONTHLY[tier_index]
    part_d_surcharge_mo = PART_D_SURCHARGES_MONTHLY[tier_index]

    # 3. Get Base Premium (Indexed value from constants dict)
    base_part_b_mo = federal_constants["base_part_b"] / 12

    # 4. Calculate Annual Total
    part_b_total_annual = 12 * persons_covered * (base_part_b_mo + part_b_surcharge_mo)
    part_d_total_annual = 12 * persons_covered * part_d_surcharge_mo
    
    return part_b_total_annual, part_d_total_annual


# --- 2. Main Orchestrator Function ---

def calculate_taxes(
    year: int,
    inflation_index: float,
    filing_status: TaxFilingStatus,
    state_of_residence: str,                 # NEW INPUT
    age1: int,
    age2: int,
    magi_two_years_ago: float,
    AGI: float,
    taxable_ordinary: float, 
    lt_cap_gains: float, 
    qualified_dividends: float,
    social_security_income: float,
    itemized_deductions_amount: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Calculates all annual taxes (Federal, State) and Medicare premiums (IRMAA).

    Returns:
        tuple[float, float, float]: (total_tax_owed, federal_tax, medicare_irmaa)
    """
    
    # 1. Fetch ALL indexed Federal constants
    constants = get_indexed_federal_constants(year, inflation_index, filing_status)

    # 2. Determine Federal Taxable Income (TI = AGI - Deduction)
    federal_deduction = constants["std_deduction"]
    
    # Add age 65+ extra deduction (indexed)
    extra_sd_count = 0
    if age1 >= 65:
        extra_sd_count += 1
    if age2 is not None and age2 >= 65:
        extra_sd_count += 1
        
    federal_deduction += extra_sd_count * constants["extra_std_deduction"]

    final_federal_deduction = max(federal_deduction, itemized_deductions_amount)
    
    # Taxable Income
    taxable_income_fed = max(0, AGI - final_federal_deduction)

    # Taxable Ordinary Base (TI minus preferential income)
    taxable_ordinary_base = max(0, taxable_income_fed - lt_cap_gains - qualified_dividends)
    
    # 3. Federal Income Tax Calculation
    federal_tax = _federal_income_tax(
        taxable_ordinary_base, 
        lt_cap_gains, 
        qualified_dividends, 
        constants
    )
    
    # 4. State Income Tax Calculation (Dispatch based on state_of_residence)
    state_tax = 0.0
    state_of_residence = state_of_residence.strip().upper()
    
    if state_of_residence == "VA":
        state_tax = _virginia_income_tax(AGI, filing_status)
    elif state_of_residence == "ME":
        state_tax = _maine_income_tax(AGI, filing_status, inflation_index)
    else:
        # Requested error/default handling
        # Logging this message will ensure it only appears once per run if necessary
        logger.warning(
            f"State Tax Calculations Not Available for '{state_of_residence}'. "
            "Defaulting to $0 state income taxes for this simulation."
        )

    # 5. Medicare IRMAA Surcharge
    part_b_total, part_d_irmaa = _get_irmaa_surcharge(
        magi_two_years_ago, 
        age1, 
        age2,
        filing_status, 
        constants
    )
    medicare_irmaa = part_b_total + part_d_irmaa

    # 6. Total Tax Owed
    total_tax_owed = federal_tax + state_tax + medicare_irmaa
    
    return total_tax_owed, federal_tax, medicare_irmaa
