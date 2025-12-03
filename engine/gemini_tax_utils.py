import numpy as np
from typing import List, Tuple

# -----------------------------------------------------------------------------
# Base Federal Brackets (2026)
# Format: (lower_bound, upper_bound, rate)
# -----------------------------------------------------------------------------
BASE_BRACKETS_FED: List[Tuple[float, float, float]] = [
    (0,      24_800, 0.10),
    (24_800, 100_800, 0.12),
    (100_800, 211_400, 0.22),
    (211_400, 403_550, 0.24),
    (403_550, 512_450, 0.32),
    (512_450, 768_700, 0.35),
    (768_700, np.inf, 0.37),
]

# Base IRMAA thresholds for MFJ (2026)
IRMAA_THRESHOLDS_MFJ = [218_000, 274_000, 342_000, 410_000, 750_000]

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def get_current_brackets(year: int, inflation_index: float = None) -> Tuple[List[Tuple[float, float, float]], List[float]]:
    """
    Return inflation-adjusted current year federal tax brackets and IRMAA thresholds.
    """
    # 1. Determine inflation factor
    if year > 2026:
        if inflation_index is not None:
            inflation_factor = inflation_index
        else:
            # Fallback (using 2% annual estimate)
            inflation_factor = (1 + 0.02) ** (year - 2026)
    else:
        inflation_factor = 1.0

    # 2. Inflate brackets and thresholds
    brackets = [(l * inflation_factor, h * inflation_factor, r) for l, h, r in BASE_BRACKETS_FED]
    thresholds = [t * inflation_factor for t in IRMAA_THRESHOLDS_MFJ]

    return brackets, thresholds


def get_tax_rate(magi: float, year: int, inflation_index: float = None) -> float:
    """
    Return marginal ordinary income tax rate for given MAGI and year
    """
    brackets, _ = get_current_brackets(year, inflation_index)
    for low, high, rate in brackets:
        if low <= magi < high:
            return rate
    return brackets[-1][2]


def get_irmaa_tier(magi_two_years_prior: float, year: int, inflation_index: float = None) -> int:
    """
    Return IRMAA tier (0-5) based on prior MAGI and year
    """
    _, thresholds = get_current_brackets(year, inflation_index)
    for tier, thresh in enumerate(thresholds):
        if magi_two_years_prior < thresh:
            return tier
    return len(thresholds)
