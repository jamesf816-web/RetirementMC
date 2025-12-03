# engine/rmd_tables.py

"""
RMD factor lookup supporting:
- 2022+ Uniform Lifetime Table
- Pre-2022 Uniform Lifetime Table (for inherited IRAs where decedent was already taking RMDs)
- SECURE Act 1.0/2.0 start ages (72 → 73 → 75)
"""

from typing import Dict

# =============================================================================
# FULL 2022+ IRS UNIFORM LIFETIME TABLE (AGES 72–120)
# =============================================================================
UNIFORM_LIFETIME_TABLE_2022: Dict[int, float] = {
    72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9,
    78: 22.0, 79: 21.1, 80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7,
    84: 16.9, 85: 16.0, 86: 15.2, 87: 14.4, 88: 13.7, 89: 12.9,
    90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5, 95: 8.9,
    96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4, 101: 6.0,
    102: 5.6, 103: 5.2, 104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1,
    108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4, 112: 3.3, 113: 3.1,
    114: 3.0, 115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3,
    120: 2.0,
}

# =============================================================================
# FULL PRE-2022 IRS UNIFORM LIFETIME TABLE (AGES 70–115)
# =============================================================================
UNIFORM_LIFETIME_TABLE_PRE2022: Dict[int, float] = {
    70: 27.4, 71: 26.5, 72: 25.6, 73: 24.7, 74: 23.8, 75: 22.9,
    76: 22.0, 77: 21.2, 78: 20.3, 79: 19.5, 80: 18.7, 81: 17.9,
    82: 17.1, 83: 16.3, 84: 15.5, 85: 14.8, 86: 14.1, 87: 13.4,
    88: 12.7, 89: 12.0, 90: 11.4, 91: 10.8, 92: 10.2, 93: 9.6,
    94: 9.1, 95: 8.6, 96: 8.1, 97: 7.6, 98: 7.1, 99: 6.7,
    100: 6.3, 101: 5.9, 102: 5.5, 103: 5.2, 104: 4.9, 105: 4.5,
    106: 4.2, 107: 3.9, 108: 3.7, 109: 3.4, 110: 3.1, 111: 2.9,
    112: 2.6, 113: 2.4, 114: 2.1, 115: 1.9,
}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def get_rmd_factor(
    age: int,
    birth_year: int | None = None,
    inherited_continuing: bool = False,
    use_pre_2022_table: bool = False,
) -> float:
    """
    Returns the IRS divisor for RMD calculations.

    Parameters
    ----------
    age : int
        Age in the distribution calendar year.
    birth_year : int | None
        Used to determine SECURE Act RMD starting age.
    inherited_continuing : bool
        True if this is an inherited IRA where the decedent was already
        taking RMDs (you must continue using PRE-2022 table).
    use_pre_2022_table : bool
        Explicit override.

    Returns
    -------
    float
        RMD divisor for the given age.
    """

    # ---------------------------------------------------------------------
    # SECURE 2.0 start-age logic (ignored for inherited accounts)
    # ---------------------------------------------------------------------
    if not inherited_continuing:
        if birth_year is not None:
            if birth_year >= 1960:
                rmd_start_age = 75
            elif 1951 <= birth_year <= 1959:
                rmd_start_age = 73
            else:
                rmd_start_age = 72

            if age < rmd_start_age:
                return 0.0

        else:
            # If no birth year is provided, conservatively assume 72+
            if age < 72:
                return 0.0

    # ---------------------------------------------------------------------
    # Choose correct table
    # ---------------------------------------------------------------------
    if inherited_continuing or use_pre_2022_table:
        table = UNIFORM_LIFETIME_TABLE_PRE2022
    else:
        table = UNIFORM_LIFETIME_TABLE_2022

    # ---------------------------------------------------------------------
    # Lookup in sorted order
    # ---------------------------------------------------------------------
    for max_age in sorted(table.keys()):
        if age <= max_age:
            return table[max_age]

    # If they somehow exceed table bounds
    return 2.0  # IRS default for 120+
    

__all__ = ["get_rmd_factor"]
