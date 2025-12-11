 engine/roth_optimizer.py

from engine.tax_planning import get_tax_planning_targets 

def optimal_roth_conversion(
    year: int, 
    inflation_index: float, 
    filing_status: str, 
    AGI_base: float, 
    traditional_balance: float, 
    roth_tax_bracket: str, 
    roth_irmaa_threshold: str
) -> float:
    """
    Calculates the optimal Roth conversion amount to fill a strategic tax bracket 
    ceiling or IRMAA threshold.
    
    Args:
        traditional_balance: Total funds available for conversion in the owner's traditional accounts.
        AGI_base: Current taxable income BEFORE this specific conversion.
    """
    
    if traditional_balance <= 0:
        return 0.0

    # 1. Get targets
    tax_target_AGI, irmaa_target_MAGI = get_tax_planning_targets(
        year=year,
        inflation_this_year=inflation_index, 
        roth_tax_bracket=roth_tax_bracket,
        roth_irmaa_threshold=roth_irmaa_threshold,
        filing_status=filing_status
    )
    
    # 2. Calculate conversion room
    # Room is the space between current income and the strategy ceiling
    room_in_tax_bracket = max(0.0, tax_target_AGI - AGI_base)
    room_in_irmaa_threshold = max(0.0, irmaa_target_MAGI - AGI_base)
    
    # The conversion is limited by:
    # A) The smallest room available (Tax vs IRMAA)
    # B) The actual money available to convert
    max_conversion_space = min(room_in_tax_bracket, room_in_irmaa_threshold)
    
    conversion = min(traditional_balance, max_conversion_space)

    # 3. Practical Rounding
    if conversion < 1000:
        return max(0.0, conversion) 
         # Let's stick to the $1k rounding for cleanliness unless it's the last bit.
        return 0.0 
    else:
        return max(0.0, round(conversion, -3))
