# engine/__init__.py  ← you can now use the clean version
from .tax_engine import calculate_taxes, get_irmaa_surcharge
from .red_tables import get_safe_withdrawal_rate, get_success_probability, apply_dynamic_rules
from .rmd_tables import get_rmd_factor          # ← now real, not stub
from .roth_optimizer import optimal_roth_conversion

__all__ = [
    "calculate_taxes",
    "get_irmaa_surcharge",
    "get_rmd_factor",
    "get_safe_withdrawal_rate",
    "get_success_probability",
    "apply_dynamic_rules",
    "RetirementSimulator",
    "optimal_roth_conversion",
]
