"""
Comprehensive U.S. tax and Medicare premium calculator for retirement planning
Tax Year 2026 (Estimated) + IRMAA (determined by 2024 MAGI)
"""

from typing import Dict, Tuple, Literal

# NOTE: The original code used external functions (get_current_brackets, get_tax_rate, get_irmaa_tier).
# To make this a self-contained, drop-in replacement, I am hardcoding the 2026
# bracket estimates directly into the relevant functions/constants.

TaxFilingStatus = Literal["single", "married_joint", "married_separate", "head_of_household"]

# =============================================================================
# 2026 Standard Deduction (Estimated based on 3% inflation from 2025 estimates)
# =============================================================================
STANDARD_DEDUCTION_2026 = {
    "single": 15050,
    "married_joint": 30100,
    "married_separate": 15050,
    "head_of_household": 22600,
}

# 2026 Extra SD (Age 65+ bonus) per spouse (Estimated)
EXTRA_STD_DEDUCTION_65 = {"single": 2000, "married_joint": 1600}

# =============================================================================
# Ordinary Income Tax Brackets 2026 (Estimated MFJ/Single)
# =============================================================================
# Brackets are (lower_bound, upper_bound, rate)
ORDINARY_BRACKETS_2026 = {
    "married_joint": [
        (0, 25544, 0.10),
        (25544, 103824, 0.12),
        (103824, 217742, 0.22),
        (217742, 415656, 0.24),
        (415656, 534467, 0.32),
        (534467, 667234, 0.35),
        (667234, float('inf'), 0.37),
    ],
    "single": [
        (0, 18000, 0.10),
        (18000, 48000, 0.12),
        (48000, 95000, 0.22),
        (95000, 185000, 0.24),
        (185000, 235000, 0.32),
        (235000, 585000, 0.35),
        (585000, float('inf'), 0.37),
    ],
}

# =============================================================================
# Long-Term Capital Gains / Qualified Dividends 2026 (Estimated)
# =============================================================================
CAPGAINS_BRACKETS_2026 = {
    "single": [(0, 48400, 0.0), (48400, 535000, 0.15), (535000, float('inf'), 0.20)],
    "married_joint": [(0, 96900, 0.0), (96900, 601300, 0.15), (601300, float('inf'), 0.20)],
}

NIIT_THRESHOLD = {"single": 200000, "married_joint": 250000}


# =============================================================================
# Social Security Taxation Thresholds (fixed)
# =============================================================================
SS_TAX_THRESHOLDS = {
    "single": [(0, 25000, 0.0), (25000, 34000, 0.50), (34000, float('inf'), 0.85)],
    "married_joint": [(0, 32000, 0.0), (32000, 44000, 0.50), (44000, float('inf'), 0.85)],
    "married_separate": [(0, 0, 0.85)],
}


# =============================================================================
# Virginia State Income Tax Brackets (Assumed unindexed top rate)
# =============================================================================
VA_TAX_BRACKETS = [
    (0, 3000, 0.02),
    (3000, 5000, 0.03),
    (5000, 17000, 0.05),
    (17000, float('inf'), 0.0575),
]
# VA Standard Deduction and Personal Exemption (using common reversion figures)
VA_SD_MFJ = 6000
VA_PE_PER_PERSON = 930
VA_PE_MFJ = VA_PE_PER_PERSON * 2


# =============================================================================
# 2026 Medicare Part B & D IRMAA (based on 2024 MAGI for a 2026 premium)
# =============================================================================
BASE_PART_B_2026 = 190.00 # Estimated 2026 Base Premium

# 2026 IRMAA Tiers (based on 2024 MAGI, using estimated 2026 dollar figures)
# IRMAA is typically based on AGI from 2 years prior.
IRMAA_TIERS_2026 = {
    "single": [
        (0,      109000),
        (109000, 137000),
        (137000, 171000),
        (171000, 205000),
        (205000, 500000),
        (500000, float('inf')),
    ],
    "married_joint": [
        (0,      218000),
        (218000, 274000),
        (274000, 342000),
        (342000, 410000),
        (410000, 750000),
        (750000, float('inf')),
    ],
}

# The additional monthly surcharge amounts (estimated for 2026)
PART_B_SURCHARGES_MONTHLY = [0.00, 72.00, 180.00, 288.00, 396.00, 432.00]
PART_D_SURCHARGES_MONTHLY = [0.00, 13.00, 34.00, 54.00, 75.00, 82.00]


# =============================================================================
# Helper Functions
# =============================================================================
def _apply_brackets(amount: float, brackets: list[tuple[float, float, float]]) -> float:
    """Apply a set of tax brackets to a given amount."""
    tax = 0.0
    for lower, upper, rate in brackets:
        if amount <= lower:
            break
        taxable = min(amount, upper) - lower
        tax += taxable * rate
        if amount <= upper:
            break
    return tax

def _find_marginal_ordinary_rate(start_income: float, segment_length: float, filing_status: str) -> float:
    """Finds the effective ordinary rate for a segment of income (used for tax differential)."""
    ordinary_brackets = ORDINARY_BRACKETS_2026.get(filing_status, ORDINARY_BRACKETS_2026["married_joint"])
    
    end_income = start_income + segment_length
    
    # We only need the rate where the segment ends, as all of the income will be in the highest bracket
    # the segment crosses.
    for lower, upper, rate in ordinary_brackets:
        if start_income >= upper:
            continue
        # If the start is in this bracket, or the end is in this bracket, this is the highest rate crossed.
        if start_income >= lower and start_income < upper:
            return rate
        if end_income > lower and end_income <= upper:
            return rate
    # Fallback to the top rate if it exceeds the highest defined bracket
    return ordinary_brackets[-1][2]


def calculate_federal_deduction(
    filing_status: TaxFilingStatus,
    age1: int = 0,
    age2: int = 0,
    itemized_deductions: float = 0,
) -> float:
    """Calculates the greater of standard deduction or itemized deductions."""
    base = STANDARD_DEDUCTION_2026.get(filing_status, 15050)
    extra = 0
    
    # 65+ bonus
    if age1 >= 65:
        # Use single/MFJ key based on filing status
        extra += EXTRA_STD_DEDUCTION_65.get(filing_status.split("_")[0], 2000)
        
    if filing_status == "married_joint" and age2 >= 65:
        extra += EXTRA_STD_DEDUCTION_65["married_joint"]
        
    std_ded = base + extra
    return max(std_ded, itemized_deductions)


def federal_income_tax(
    taxable_ordinary: float,
    lt_cap_gains: float,
    qual_dividends: float,
    filing_status: str,
) -> float:
    # 1. Setup Brackets
    ordinary_brackets = ORDINARY_BRACKETS_2026.get(filing_status, ORDINARY_BRACKETS_2026["married_joint"])
    cap_gains_brackets = CAPGAINS_BRACKETS_2026.get(filing_status, CAPGAINS_BRACKETS_2026["married_joint"])
    
    preferential_income = lt_cap_gains + qual_dividends
    ordinary_tax_base = taxable_ordinary
    total_ti = ordinary_tax_base + preferential_income

    # --- FIX: Use the robust subtraction method for correct stacking ---
    
    # 1. Calculate the hypothetical tax on the TOTAL Taxable Income (TI) if it were ALL ordinary income.
    tax_on_ti_at_ord_rates = _apply_brackets(total_ti, ordinary_brackets)
    
    # 2. Calculate the tax due on the Preferential Income (LTCG/QDiv)
    tax_on_preferential_income = 0.0
    
    # The LTCG/QDiv income starts stacking immediately after the ordinary tax base ends.
    cumulative_income = ordinary_tax_base 
    remaining_preferential = preferential_income
    
    # Calculate the discount gained by taxing preferential income at special rates.
    total_ordinary_tax_on_preferential = 0.0
    
    for lower, upper, rate in cap_gains_brackets:
        if cumulative_income >= upper:
            continue
            
        ltcg_bracket_start = max(lower, cumulative_income)
        
        # Amount of preferential income that falls into this bracket
        taxable_at_this_rate = min(upper, cumulative_income + remaining_preferential) - ltcg_bracket_start
        
        if taxable_at_this_rate > 0:
            # Add the tax at the special rate
            tax_on_preferential_income += taxable_at_this_rate * rate
            
            # Find the marginal ordinary rate for this specific segment
            ord_rate = _find_marginal_ordinary_rate(cumulative_income, taxable_at_this_rate, filing_status)
            total_ordinary_tax_on_preferential += taxable_at_this_rate * ord_rate
            
            remaining_preferential -= taxable_at_this_rate
            cumulative_income = upper
            
            if remaining_preferential <= 0:
                break

    # Total Federal Tax (before NIIT) = Total Ordinary Tax - (Ordinary Tax on Pref Income - Actual Tax on Pref Income)
    tax_before_niit = tax_on_ti_at_ord_rates - total_ordinary_tax_on_preferential + tax_on_preferential_income

    # 3. Net Investment Income Tax (NIIT)
    niit_threshold = NIIT_THRESHOLD.get(filing_status, 250000)
    
    # AGI proxy for NIIT (Assumed to be total Gross Income - Adjustments)
    magi_proxy = total_ti + calculate_federal_deduction(filing_status) # Approx AGI
    
    # The lesser of (3.8% of income over threshold) OR (3.8% of net investment income)
    income_over_threshold = max(0, magi_proxy - niit_threshold)
    niit = min(income_over_threshold, lt_cap_gains + qual_dividends) * 0.038
    
    # 4. Total Federal Tax
    return tax_before_niit + niit


def taxable_social_security(ss_benefit: float, combined_income: float, filing_status: TaxFilingStatus) -> float:
    """Calculates the taxable portion of Social Security benefit."""
    if ss_benefit <= 0:
        return 0.0
    thresholds = SS_TAX_THRESHOLDS[filing_status]
    
    taxable = 0.0
    
    # Simpler logic for calculating amount subject to tax:
    if filing_status == "married_joint":
        if combined_income <= 32000:
             taxable = 0.0
        elif combined_income <= 44000: # 50% segment
             taxable = (combined_income - 32000) * 0.5
        else: # 85% segment
             taxable = (44000 - 32000) * 0.5 + (combined_income - 44000) * 0.85
    elif filing_status == "single":
        if combined_income <= 25000:
             taxable = 0.0
        elif combined_income <= 34000: # 50% segment
             taxable = (combined_income - 25000) * 0.5
        else: # 85% segment
             taxable = (34000 - 25000) * 0.5 + (combined_income - 34000) * 0.85
    else: # married_separate or head_of_household (simplified)
         taxable = combined_income * 0.85

    # Cap the result at 85% of the total benefit
    return min(taxable, ss_benefit * 0.85)


def get_irmaa_surcharge(magi_two_years_ago: float, filing_status: TaxFilingStatus, persons) -> Tuple[float, float]:
    """Returns (annual Part B total, annual Part D IRMAA only) for household."""

    # Determine the IRMAA tier based on MAGI two years prior (2024 MAGI for 2026 premium)
    tiers = IRMAA_TIERS_2026.get(filing_status, IRMAA_TIERS_2026[filing_status])
    tier_index = 0
    for i, (lower, upper) in enumerate(tiers):
        if magi_two_years_ago < upper:
            tier_index = i
            break
        tier_index = i # handles the 'inf' case

    # Map tier index to the monthly surcharge amounts
    b_addl = PART_B_SURCHARGES_MONTHLY[tier_index]
    d_addl = PART_D_SURCHARGES_MONTHLY[tier_index]

    # Calculate annual cost
    part_b_annual = (BASE_PART_B_2026 + b_addl) * 12 * persons
    part_d_annual = d_addl * 12 * persons

    return part_b_annual, part_d_annual

def virginia_income_tax(taxable_income: float, inflation_index: float) -> float:
    """Compute Virginia state income tax with inflation adjustment."""
    # Since VA is known for having mostly static brackets, we apply inflation 
    # to the bounds only if an index > 1 is provided.
    VA_BRACKETS_ADJUSTED = [
        (low * inflation_index, high * inflation_index if high != float('inf') else float('inf'), rate)
        for low, high, rate in VA_TAX_BRACKETS
    ]
    return _apply_brackets(taxable_income, VA_BRACKETS_ADJUSTED)


# =============================================================================
# Public API expected by your simulator
# =============================================================================
def calculate_taxes(
    ordinary_income: float,
    ss_benefit: float = 0,
    lt_cap_gains: float = 0,
    qualified_dividends: float = 0,
    filing_status: TaxFilingStatus = "married_joint",
    age1: int = 65,
    age2: int = 65,
    magi_two_years_ago: float = 0,
    itemized_deductions: float = 0,
    year: int = 2026, # Default to the new year
    inflation_index: float = 1.0,

) -> Dict[str, float]:
    
    # 1. Taxable SS
    # Combined income for SS is AGI (excl SS) + 50% of SS benefit
    combined = ordinary_income + qualified_dividends + lt_cap_gains + ss_benefit / 2
    taxable_ss = taxable_social_security(ss_benefit, combined, filing_status)

    # 2. Total AGI
    AGI = ordinary_income + lt_cap_gains + qualified_dividends + taxable_ss
    
    # 3. Federal Taxable Income
    federal_deduction = calculate_federal_deduction(filing_status, age1, age2, itemized_deductions)
    
    # Taxable Income = AGI - Deduction
    taxable_income_fed = max(0, AGI - federal_deduction)

    # Taxable Ordinary Base (Taxable Income minus preferential income)
    taxable_ordinary_base = max(0, taxable_income_fed - lt_cap_gains - qualified_dividends)

    # 4. Federal tax
    # We pass the Taxable Ordinary Base (TI - LTCG - QDiv) to the tax function
    fed_tax = federal_income_tax(
        taxable_ordinary_base, 
        lt_cap_gains, 
        qualified_dividends, 
        filing_status
    )

    # 5. Medicare (IRMAA is based on MAGI from 2 years prior)
    # Note: 'year' is ignored in the helper as constants are hardcoded for 2026
    persons = 2
    if age1 < 65:
        persons = 1
    
    part_b_total, part_d_irmaa = get_irmaa_surcharge(magi_two_years_ago, filing_status, persons)

    # 6. Virginia state tax (FIXED: Uses AGI and VA-specific deductions/exemptions)
    if filing_status == "married_joint":
        va_deduction = VA_SD_MFJ + VA_PE_MFJ
    else:
        # Simplified assumption for other statuses for this example
        va_deduction = 7800 + 930 # Single estimate
        
    taxable_income_va = max(0, AGI - va_deduction)
    va_tax = virginia_income_tax(taxable_income_va, inflation_index)

    return {
        "federal_tax": round(fed_tax, 2),
        "taxable_ss": round(taxable_ss, 2),
        "medicare_part_b": round(part_b_total, 2),
        "medicare_part_d_irmaa": round(part_d_irmaa, 2),
        "total_medicare": round(part_b_total + part_d_irmaa, 2),
        "state_tax_va": round(va_tax, 2),
    }
