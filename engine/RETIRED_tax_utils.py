# engine/tax_utils.py
import numpy as np
from typing import List, Tuple, Dict

# -----------------------------------------------------------------------------
# Base Federal Brackets (2026) - Married Filing Jointly
# -----------------------------------------------------------------------------
BASE_BRACKETS_FED: List[Tuple[float, float, float]] = [
    (0,        24_800,  0.10),
    (24_800,  100_800, 0.12),
    (100_800, 211_400, 0.22),
    (211_400, 403_550, 0.24),
    (403_550, 512_450, 0.32),
    (512_450, 768_700, 0.35),
    (768_700, np.inf,    0.37),
]

IRMAA_THRESHOLDS_MFJ = [218_000, 274_000, 342_000, 410_000, 750_000]

def get_current_brackets(year: int, inflation_index: float = None):
    """
    Returns inflation-adjusted brackets in TWO formats for maximum compatibility:

    Returns:
        brackets_list:   List[Tuple[low, high, rate]]     ← old code expects this
        brackets_dict:   Dict[str, List[float]]           ← new optimizer-friendly format
        irmaa_thresholds: List[float]
    """
    # 1. Inflation factor
    if year > 2026:
        if inflation_index is not None:
            inflation_factor = inflation_index
        else:
            # Deterministic inflation per year (remove randomness in production if unwanted)
            rng = np.random.default_rng(seed=year)
            rates = rng.normal(0.02, 0.01, year - 2026)
            inflation_factor = np.prod(1 + rates)
    else:
        inflation_factor = 1.0

    # 2. Build inflated list of tuples (preserves order!)
    brackets_list: List[Tuple[float, float, float]] = [
        (low * inflation_factor, high * inflation_factor if np.isfinite(high) else np.inf, rate)
        for low, high, rate in BASE_BRACKETS_FED
    ]

    # 3. Also build the dictionary format (used by optimizer/solver)
    brackets_dict: Dict[str, List[float]] = {}
    for low, high, rate in BASE_BRACKETS_FED:
        key = f"{int(rate * 100)}_percent"
        brackets_dict[key] = [
            low * inflation_factor,
            high * inflation_factor if np.isfinite(high) else np.inf,
        ]

    # 4. IRMAA thresholds
    irmaa_thresholds = [t * inflation_factor for t in IRMAA_THRESHOLDS_MFJ]

    return brackets_list, brackets_dict, irmaa_thresholds


# Fixed helper functions
def get_tax_rate(magi: float, year: int, inflation_index: float = None) -> float:
    brackets_list, _, _ = get_current_brackets(year, inflation_index)
    for low, high, rate in brackets_list:
        if low <= magi < high:
            return rate
    return 0.37  # top bracket


def get_irmaa_tier(magi_two_years_prior: float, year: int, inflation_index: float = None) -> int:
    _, _, thresholds = get_current_brackets(year, inflation_index)
    for tier, thresh in enumerate(thresholds):
        if magi_two_years_prior < thresh:
            return tier
    return len(thresholds)  # tier 5
