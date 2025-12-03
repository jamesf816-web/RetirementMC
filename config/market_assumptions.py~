# =============================================================================
# USER INPUT SECTION - EDIT THIS PART WITH YOUR DATA
# =============================================================================
import numpy as np

# Basic info
current_year = 2026
current_age_JEF = 60
current_age_SEF = 66
retirement_age = 60
end_age = 100
n_years = end_age - current_age_JEF
n_simulations = 1000

# Inflation & market regime (November 2025 reality)
initial_inflation_mu = 0.035
initial_inflation_sigma = 0.025
long_term_inflation_mu = 0.025
long_term_inflation_sigma = 0.015
years_to_revert = 10

initial_equity_mu = 0.06
initial_equity_sigma = 0.16
long_term_equity_mu = 0.075
long_term_equity_sigma = 0.165

initial_bond_mu = 0.03
initial_bond_sigma = 0.08
long_term_bond_mu = 0.045
long_term_bond_sigma = 0.09

corr_matrix = np.array([
    [ 1.00, -0.30,  0.20],
    [-0.30,  1.00, -0.50],
    [ 0.20, -0.50,  1.00]
])

# Accounts (2025 ending balances - used to determine 2026 RMDs)
accounts = {
    "GST_Exempt":     {"balance": 3_132_760, "equity": 0.75, "bond": 0.25, "tax": "trust",      "owner": "JEF", "mandatory_yield": 0.015},
    "REF_Irrev":      {"balance": 1_806_065, "equity": 0.75, "bond": 0.25, "tax": "trust",      "owner": "JEF", "mandatory_yield": 0.015},
    "JEF_Taxable":    {"balance": 1_347_857, "equity": 0.92, "bond": 0.08, "tax": "taxable",    "owner": "JEF", "basis": 789_874},
    "Joint_Taxable":  {"balance":   857_710, "equity": 0.26, "bond": 0.74, "tax": "taxable",    "owner": "JEF", "basis": 800_000},
    "JEF_RolloverIRA":{"balance":   854_793, "equity": 0.77, "bond": 0.23, "tax": "traditional","owner": "JEF"},
    "JEF_TradIRA":    {"balance":   255_797, "equity": 0.82, "bond": 0.18, "tax": "traditional","owner": "JEF"},
    "Inherited_IRA63":{"balance":   915_953, "equity": 0.81, "bond": 0.19, "tax": "inherited",  "owner": "JEF", "rmd_factor_table": "IRS_Single_Life"},
    "Inherited_IRA77":{"balance":   175_130, "equity": 0.78, "bond": 0.22, "tax": "inherited",  "owner": "JEF", "rmd_factor_table": "IRS_Single_Life"},
    "JEF_TIAA401K":   {"balance":   367_118, "equity": 0.81, "bond": 0.19, "tax": "traditional","owner": "JEF"},
    "SEF_TradIRA":    {"balance":   212_170, "equity": 1.00, "bond": 0.00, "tax": "traditional","owner": "SEF"},
    "JEF_RothIRA":    {"balance":   172_451, "equity": 1.00, "bond": 0.00, "tax": "roth",       "owner": "JEF"},
    "SEF_RothIRA":    {"balance":    60_726, "equity": 0.85, "bond": 0.15, "tax": "roth",       "owner": "SEF"},
}

# Income streams
deferred_salary = [
    {"year": 2026, "amount": 6_250},
    {"year": 2027, "amount": 6_250},
    {"year": 2028, "amount": 6_250},
    {"year": 2029, "amount": 6_250},
    {"year": 2030, "amount": 6_250},
]

social_security = {
    "SEF": {"start_age": 67, "annual_pia": 9_297},
    "JEF": {"start_age": 70, "annual_pia": 78_858},
    "cola": True
}

pension = {
    "JEF": {"start_age": 65, "annual_amount": 31_876},
    "cola": False
}

# Expenses
medicare_start_age = 65
medicare_supplement_annual = 5_000
mortgage_payoff_year = 2031
mortgage_monthly_until_payoff = 3_625 * 12
taxes_and_insurance = 3_421 * 2 + 5_000 #guessing what real homeowners will cost

car_replacement_cycle = 4
car_cost_today = 50_000
car_inflation = 0.04

home_repair_prob = 0.1
home_repair_mean = 25_000
home_repair_shape = 2.0

#gifting = 40_000
#travel = 50_000

# Tax and withdrawal strategy (2026 brackets and threshold)
withdrawal_floor = 175_000
withdrawal_ceiling = 600_000
