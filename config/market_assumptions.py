# =============================================================================
# Market Info used in simulations
# =============================================================================
import numpy as np

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
