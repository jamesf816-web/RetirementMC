# config/expense_assumptions.py
# These are **reasonable defaults** — user can override in GUI

# Healthcare (generated in code, but defaults here for clarity)
medicare_start_age = 65
medicare_part_b_base_2026 = 185.00 * 12      # will be inflation-adjusted
medicare_supplement_annual = 404.70 * 12     # 2026 total $589.70
irmaa_brackets_start_year = 2026

# Housing
mortgage_payoff_year = 2031
property_tax_and_insurance = (3421.00 * 2) + 4800.00 # bi-annual taxes plus gross estimate of new homeowners policy
mortgage_monthly_until_payoff = 3756.47 - (property_tax_and_insurance / 12) # wantjust principle and interest here

# Transportation
car_replacement_cycle = 8
car_cost_today = 55_000
car_inflation = 0.04

# Home Maintenance (lumpy)
home_repair_prob= 0.12
home_repair_mean = 28_000
home_repair_shape = 2.0   # lognormal shape

# One-time or irregular (user can add more in GUI later)
lumpy_expenses = [
    # {"year": 2030, "description": "New roof", "amount": 35_000},
    # {"year": 2035, "description": "HVAC replacement", "amount": 18_000},
]
