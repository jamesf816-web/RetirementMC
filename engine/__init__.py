# engine/__init__.py

# Only expose the main tax calculation function, which orchestrates all helpers.
from .tax_engine import calculate_taxes 

# Expose the simulator (for the callbacks file)
from .simulator import RetirementSimulator

# You may also want to expose other core engines if they are used externally:
# from .roth_optimizer import optimal_roth_conversion 
# from .tax_planning import get_tax_planning_targets
