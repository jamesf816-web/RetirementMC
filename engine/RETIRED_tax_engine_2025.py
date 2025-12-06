# engine/tax_engine.py
"""
Comprehensive U.S. tax and Medicare premium calculator for retirement planning
Tax Year 2025 + 2025 IRMAA (determined by 2023 MAGI)
"""

from typing import Dict, Tuple, Literal
from engine.tax_utils import get_current_brackets, get_tax_rate, get_irmaa_tier

TaxFilingStatus = Literal["single", "married_joint", "married_separate", "head_of_household"]

# =============================================================================
# 2025 Standard Deduction + Age 65+ bonus
# =============================================================================
STANDARD_DEDUCTION_2025 = {
    "single": 14600,
    "married_joint": 29200,
    "married_separate": 14600,
    "head_of_household": 21900,
}

EXTRA_STD_DEDUCTION_65 = {"single": 1950, "married_joint": 1550}  # per spouse for MFJ


# =============================================================================
# Long-Term Capital Gains / Qualified Dividends 2025
# =============================================================================
CAPGAINS_BRACKETS_2025 = {
    "single": [(0, 47025, 0.0), (47025, 518900, 0.15), (518900, float('inf'), 0.20)],
    "married_joint": [(0, 94050, 0.0), (94050, 583750, 0.15), (583750, float('inf'), 0.20)],
}

NIIT_THRESHOLD = {"single": 200000, "married_joint": 250000}


# =============================================================================
# Social Security Taxation Thresholds (fixed the broken line!)
# =============================================================================
SS_TAX_THRESHOLDS = {
    "single": [(0, 25000, 0.0), (25000, 34000, 0.50), (34000, float('inf'), 0.85)],
    "married_joint": [(0, 32000, 0.0), (32000, 44000, 0.50), (44000, float('inf'), 0.85)],
    "married_separate": [(0, 0, 0.85)],  # almost always 85% if lived together anytime during year
}


# =============================================================================
# 2025 Medicare Part B & D IRMAA (based on 2023 MAGI)
# =============================================================================
BASE_PART_B_2025 = 185.00

IRMAA_2025_PART_B = {  # additional monthly amount per person
    "single": [
        (0,      103000,   0.00),
        (103000, 129000,  69.90),
        (129000, 161000, 174.80),
        (161000, 193000, 279.70),
        (193000, 500000, 384.60),
        (500000, float('inf'), 419.30),
    ],
    "married_joint": [
        (0,      206000,   0.00),
        (206000, 258000,  69.90),
        (258000, 322000, 174.80),
        (322000, 386000, 279.70),
        (386000, 750000, 384.60),
        (750000, float('inf'), 419.30),
    ],
}

IRMAA_2025_PART_D = {  # additional monthly amount per person
    "single": [
        (0,      103000,   0.00),
        (103000, 129000,  12.90),
        (129000, 161000,  33.30),
        (161000, 193000,  53.80),
        (193000, 500000,  74.20),
        (500000, float('inf'),  81.00),
    ],
    "married_joint": [
        (0,      206000,   0.00),
        (206000, 258000,  12.90),
        (258000, 322000,  33.30),
        (322000, 386000,  53.80),
        (386000, 750000,  74.20),
        (750000, float('inf'),  81.00),
    ],
}


# =============================================================================
# Helper Functions
# =============================================================================
def _apply_brackets(amount: float, brackets: list[tuple[float, float, float]]) -> float:
    """
    Apply a set of tax brackets to a given amount.
    
    Args:
        amount: Taxable income
        brackets: List of tuples (lower_bound, upper_bound, rate)
    
    Returns:
        Total tax owed
    """
    tax = 0.0
    for lower, upper, rate in brackets:
        if amount <= lower:
            break
        # Tax the portion of income within this bracket
        taxable = min(amount, upper) - lower
        tax += taxable * rate
        if amount <= upper:
            break
    return tax


def calculate_taxable_income(
    gross_ordinary: float,
    filing_status: TaxFilingStatus,
    age1: int = 0,
    age2: int = 0,
    itemized_deductions: float = 0,
) -> float:
    base = STANDARD_DEDUCTION_2025.get(filing_status, 14600)
    extra = 0
    if age1 >= 65:
        extra += EXTRA_STD_DEDUCTION_65.get(filing_status.split("_")[0], 1950)
    if filing_status == "married_joint" and age2 >= 65:
        extra += 1550
    std_ded = base + extra
    return max(0, gross_ordinary - max(std_ded, itemized_deductions))

def federal_income_tax(
    taxable_ordinary: float,
    lt_cap_gains: float,
    qual_dividends: float, # Note: Since you have no QDiv, we treat it like LTCG for now
    filing_status: str,
    year: int,
    inflation_index: float = None
) -> float:
    # Get brackets (assuming brackets includes ordinary and LTCG rates)
    ordinary_brackets, _ = get_current_brackets(year, inflation_index)
    
    # Use the hard-coded 2025 cap gains brackets as a template for now
    cap_gains_brackets = CAPGAINS_BRACKETS_2025.get(filing_status, CAPGAINS_BRACKETS_2025["married_joint"])
    
    # 1. Base Tax on Ordinary Income
    # We must treat LTCG/QDiv as the "top" layer of income.
    
    # Taxable income before adding LTCG/QDiv
    ordinary_tax_base = max(0, taxable_ordinary - lt_cap_gains - qual_dividends)
    
    # Calculate tax on the ordinary income portion
    tax_on_ordinary = _apply_brackets(ordinary_tax_base, ordinary_brackets)
    
    # 2. Tax on Preferential Income (LTCG/QDiv)
    # The LTCG/QDiv is taxed based on where it falls on top of the Ordinary Tax Base.
    
    preferential_income = lt_cap_gains + qual_dividends
    tax_on_preferential = 0.0
    
    # Start checking the LTCG brackets where the ordinary income left off
    cumulative_income = ordinary_tax_base
    remaining_ltcg = preferential_income
    
    # 

    for lower, upper, rate in cap_gains_brackets:
        # Check if the cumulative income has already surpassed this LTCG bracket
        if cumulative_income >= upper:
            continue
            
        # Determine where the LTCG bracket starts in relation to the ordinary income base
        ltcg_bracket_start = max(lower, cumulative_income)
        
        # Determine the maximum amount of income that can be taxed at this LTCG rate
        # This is the lesser of: the top of the bracket, or the top of the LTCG income
        taxable_at_this_rate = min(upper, cumulative_income + remaining_ltcg) - ltcg_bracket_start
        
        if taxable_at_this_rate > 0:
            tax_on_preferential += taxable_at_this_rate * rate
            remaining_ltcg -= taxable_at_this_rate
            
        if remaining_ltcg <= 0:
            break
            
        # Update cumulative income to the top of the bracket for the next iteration
        cumulative_income = upper

    # 3. Net Investment Income Tax (NIIT)
    niit_threshold = NIIT_THRESHOLD.get(filing_status, 250000)
    # AGI proxy for NIIT (This is simplified, AGI is generally Gross Income - Adjustments)
    magi_proxy = taxable_ordinary + lt_cap_gains + qual_dividends 
    
    # The lesser of (3.8% of income over threshold) OR (3.8% of net investment income)
    income_over_threshold = max(0, magi_proxy - niit_threshold)
    niit = min(income_over_threshold, lt_cap_gains + qual_dividends) * 0.038
    
    # 4. Total Federal Tax
    return tax_on_ordinary + tax_on_preferential + niit

def taxable_social_security(ss_benefit: float, combined_income: float, filing_status: TaxFilingStatus) -> float:
    if ss_benefit <= 0:
        return 0.0
    thresholds = SS_TAX_THRESHOLDS[filing_status]
    taxable = 0.0
    for low, high, rate in thresholds:
        if combined_income <= low:
            break
        segment = min(combined_income - low, high - low if high != float('inf') else combined_income - low)
        taxable += rate * min(segment, ss_benefit - taxable)
    return min(taxable, ss_benefit * 0.85)


def get_irmaa_surcharge(magi_two_years_ago: float, filing_status: TaxFilingStatus, year) -> Tuple[float, float]:
    """Returns (annual Part B total, annual Part D IRMAA only) for household"""
    persons = 2 if filing_status == "married_joint" else 1

    # Part B
    tier = get_irmaa_tier(magi_two_years_ago, year)
    PART_B_SURCHARGES = [0.0, 69.9, 174.8, 279.7, 384.6, 419.3]
    # Map tier to monthly Part B / D surcharges
    PART_D_SURCHARGES = [0.0, 12.9, 33.3, 53.8, 74.2, 81.0]

    b_addl = PART_B_SURCHARGES[tier]
    d_addl = PART_D_SURCHARGES[tier]

    part_b_annual = (BASE_PART_B_2025 + b_addl) * 12 * persons
    part_d_annual = d_addl * 12 * persons

    return part_b_annual, part_d_annual

# =============================================================================
# Virginia State Income Tax Brackets 2025
# =============================================================================
VA_TAX_BRACKETS_2025 = [
    (0, 3000, 0.02),
    (3000, 5000, 0.03),
    (5000, 17000, 0.05),
    (17000, float('inf'), 0.0575),
]
def virginia_income_tax(taxable_income: float, inflation_index: float) -> float:
    """Compute Virginia state income tax with inflation adjustment."""
    VA_TAX_BRACKETS = [
        (low * inflation_index, high * inflation_index if high != float('inf') else float('inf'), rate)
        for low, high, rate in VA_TAX_BRACKETS_2025
    ]
    return _apply_brackets(taxable_income, VA_TAX_BRACKETS)


# =============================================================================
# Public API expected by your simulator
# =============================================================================
def calculate_taxes(
    ordinary_income: float,
    ss_benefit: float = 0,
    lt_cap_gains: float = 0,
    qualified_dividends: float = 0,
    filing_status: str = "married_joint",
    age1: int = 65,
    age2: int = 65,
    magi_two_years_ago: float = 0,
    itemized_deductions: float = 0,
    year: int = 2025,
    inflation_index: float = 1,

) -> Dict[str, float]:
    # 1. Taxable SS
    combined = ordinary_income + qualified_dividends + ss_benefit / 2
    taxable_ss = taxable_social_security(ss_benefit, combined, filing_status)

    # 2. Taxable ordinary income
    gross_ordinary = ordinary_income + taxable_ss
    taxable_ordinary = calculate_taxable_income(gross_ordinary, filing_status, age1, age2, itemized_deductions)

    # 3. Federal tax
    fed_tax = federal_income_tax(taxable_ordinary, lt_cap_gains, qualified_dividends, filing_status, year, inflation_index)

    # 4. Medicare
    part_b_total, part_d_irmaa = get_irmaa_surcharge(magi_two_years_ago, filing_status, year)

    # 5. Virginia state tax
    va_tax = virginia_income_tax(taxable_ordinary, inflation_index)

    return {
        "federal_tax": round(fed_tax, 2),
        "taxable_ss": round(taxable_ss, 2),
        "medicare_part_b": round(part_b_total, 2),
        "medicare_part_d_irmaa": round(part_d_irmaa, 2),
        "total_medicare": round(part_b_total + part_d_irmaa, 2),
        "state_tax_va": round(va_tax, 2),
 
    }
