# utils/tax_utils.py
import numpy as np
from typing import List, Tuple, Dict, Literal, Union

# Define the acceptable set of filing statuses for type hinting
TaxFilingStatus = Literal["single", "married_filing_jointly", "married_separate", "head_of_household"]
BASE_YEAR = 2026 # Base year for all estimated Federal constants

# =============================================================================
# 1. Federal Ordinary Income Tax Brackets (2026 Estimated)
# =============================================================================

ORDINARY_BRACKETS_2026: Dict[TaxFilingStatus, List[Tuple[float, float, float]]] = {
    "married_filing_jointly": [
        (0, 24_800, 0.10), (24_800, 100_800, 0.12), (100_800, 211_400, 0.22), 
        (211_400, 403_550, 0.24), (403_550, 512_450, 0.32), (512_450, 768_700, 0.35), 
        (768_700, np.inf, 0.37),
    ],
    "single": [
        (0, 12_400, 0.10), (12_400, 50_400, 0.12), (50_400, 110_650, 0.22), 
        (110_650, 196_150, 0.24), (196_150, 250_000, 0.32), (250_000, 622_050, 0.35), 
        (622_050, np.inf, 0.37),
    ],
    "head_of_household": [
        (0, 18_600, 0.10), (18_600, 72_000, 0.12), (72_000, 148_000, 0.22), 
        (148_000, 258_000, 0.24), (258_000, 321_450, 0.32), (321_450, 622_050, 0.35), 
        (622_050, np.inf, 0.37),
    ],
    "married_separate": [
        (0, 12_400, 0.10), (12_400, 50_400, 0.12), (50_400, 105_700, 0.22), 
        (105_700, 201_775, 0.24), (201_775, 256_225, 0.32), (256_225, 384_350, 0.35), 
        (384_350, np.inf, 0.37),
    ]
}

# =============================================================================
# 2. Federal Preferential Income Tax Brackets (Capital Gains / QDivs)
# =============================================================================
CAPGAINS_BRACKETS_2026: Dict[TaxFilingStatus, List[Tuple[float, float, float]]] = {
    "single": [(0, 48400, 0.0), (48400, 535000, 0.15), (535000, np.inf, 0.20)],
    "married_filing_jointly": [(0, 96900, 0.0), (96900, 601300, 0.15), (601300, np.inf, 0.20)],
    "married_separate": [(0, 48450, 0.0), (48450, 300650, 0.15), (300650, np.inf, 0.20)],
    "head_of_household": [(0, 72900, 0.0), (72900, 568300, 0.15), (568300, np.inf, 0.20)],
}

# =============================================================================
# 3. Federal Deduction, Exemption, and Surcharge Thresholds (Indexed)
# =============================================================================
STANDARD_DEDUCTION_2026: Dict[TaxFilingStatus, float] = {
    "single": 15050,
    "married_filing_jointly": 30100,
    "married_separate": 15050,
    "head_of_household": 22600,
}

EXTRA_STD_DEDUCTION_65: Dict[str, float] = {"single": 2000, "married_filing_jointly": 1600}

NIIT_THRESHOLD_2026: Dict[str, float] = {"single": 200000, "married_filing_jointly": 250000}

IRMAA_THRESHOLDS_2026: Dict[str, List[float]] = {
    "married_filing_jointly": [218_000, 274_000, 342_000, 410_000, 750_000],
    "single": [109_000, 137_000, 171_000, 205_000, 410_000],
    "head_of_household": [109_000, 137_000, 171_000, 205_000, 410_000],
    "married_separate": [109_000, 137_000, 171_000, 205_000, 410_000],
}


# =============================================================================
# 4. Fixed / Non-Indexed Federal Tax Parameters 
# =============================================================================

# Social Security Taxation Thresholds (Statutory and NOT indexed)
SS_TAX_THRESHOLDS: Dict[str, List[Tuple[float, float, float]]] = {
    "single": [(0, 25000, 0.0), (25000, 34000, 0.50), (34000, np.inf, 0.85)],
    "married_filing_jointly": [(0, 32000, 0.0), (32000, 44000, 0.50), (44000, np.inf, 0.85)],
    "married_separate": [(0, 0, 0.85)],
}

# Medicare Surcharges (Monthly) - BASE PREMIUM IS INDEXED IN FUNCTION
BASE_PART_B_2026 = 190.00
PART_B_SURCHARGES_MONTHLY = [0.00, 72.00, 180.00, 288.00, 396.00, 432.00]
PART_D_SURCHARGES_MONTHLY = [0.00, 13.00, 34.00, 54.00, 75.00, 82.00]


# =============================================================================
# 5. State Tax Parameters (Generally non-indexed, or indexed separately)
# =============================================================================

# VIRGINIA (VA) Constants
VA_TAX_BRACKETS = [
    (0, 3000, 0.02), (3000, 5000, 0.03), (5000, 17000, 0.05), (17000, np.inf, 0.0575),
]
VA_SD_MFJ = 6000 # Virginia Standard Deduction MFJ
VA_PE_PER_PERSON = 930 # Virginia Personal Exemption per person

# MAINE (ME) Constants (Based on 2024/2025 estimates for a 2026 baseline)
ME_TAX_BRACKETS: Dict[TaxFilingStatus, List[Tuple[float, float, float]]] = {
    "single": [(0, 6300, 0.058), (6300, 50350, 0.0675), (50350, np.inf, 0.0715)],
    "married_filing_jointly": [(0, 12600, 0.058), (12600, 100700, 0.0675), (100700, np.inf, 0.0715)],
    # Use Single for MFS and HOH as simplified placeholder if official HOH/MFS brackets are not defined
    "married_separate": [(0, 6300, 0.058), (6300, 50350, 0.0675), (50350, np.inf, 0.0715)],
    "head_of_household": [(0, 6300, 0.058), (6300, 50350, 0.0675), (50350, np.inf, 0.0715)],
}

# Maine Standard Deduction (Indexed in tax_engine, assumed 2026 baseline)
ME_SD_2026: Dict[TaxFilingStatus, float] = {
    "single": 13850,
    "married_filing_jointly": 27700,
    "married_separate": 13850,
    "head_of_household": 20800,
}


# =============================================================================
# 6. Core Utility Function (Returns all indexed Federal values)
# =============================================================================

def get_indexed_federal_constants(year: int, inflation_index: float, filing_status: TaxFilingStatus) -> Dict[str, Union[float, List, Dict]]:
    """
    Returns a dictionary of all Federal tax brackets, deductions, and thresholds 
    indexed to the current simulation year.
    """
    
    # Determine inflation factor: 1.0 if before BASE_YEAR, else cumulative index
    inflation_factor = inflation_index if year > BASE_YEAR else 1.0

    # Get Base Data
    base_ord_brackets = ORDINARY_BRACKETS_2026.get(filing_status, ORDINARY_BRACKETS_2026["married_filing_jointly"])
    base_cg_brackets = CAPGAINS_BRACKETS_2026.get(filing_status, CAPGAINS_BRACKETS_2026["married_filing_jointly"])
    base_irmaa = IRMAA_THRESHOLDS_2026.get(filing_status, IRMAA_THRESHOLDS_2026["married_filing_jointly"])
    
    # Build Indexed Brackets and Tiers
    def _index_brackets(base_brackets: List[Tuple[float, float, float]]):
        """Helper to index bracket bounds."""
        indexed_list = []
        indexed_dict = {}
        for low, high, rate in base_brackets:
            inflated_low = low * inflation_factor
            inflated_high = high * inflation_factor if np.isfinite(high) else np.inf
            indexed_list.append((inflated_low, inflated_high, rate))
            indexed_dict[f"{int(rate * 100)}_percent"] = [inflated_low, inflated_high] 
        return indexed_list, indexed_dict

    ord_list, ord_dict = _index_brackets(base_ord_brackets)
    cg_list, cg_dict = _index_brackets(base_cg_brackets)
    
    indexed_irmaa = [t * inflation_factor for t in base_irmaa]
    indexed_base_part_b = BASE_PART_B_2026 * inflation_factor
    
    # Index Deductions and Thresholds
    indexed_std_deduction = STANDARD_DEDUCTION_2026.get(filing_status, 0.0) * inflation_factor
    indexed_extra_std = EXTRA_STD_DEDUCTION_65.get(filing_status, EXTRA_STD_DEDUCTION_65["married_filing_jointly"]) * inflation_factor
    indexed_niit = NIIT_THRESHOLD_2026.get("married_filing_jointly" if filing_status in ["married_filing_jointly", "married_separate"] else "single") * inflation_factor
    
    # Return Comprehensive Dictionary
    return {
        "ord_list": ord_list,
        "ord_dict": ord_dict, # For planning targets
        "cg_list": cg_list,
        "std_deduction": indexed_std_deduction,
        "extra_std_deduction": indexed_extra_std,
        "niit_threshold": indexed_niit,
        "irmaa_thresholds": indexed_irmaa,
        "base_part_b": indexed_base_part_b,
    }
