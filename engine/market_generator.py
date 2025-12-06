# market_generator.py
#
# This code generates market retuns and inflation
# It explicitly takes into account near term (10 year) correlations to current
# (2025) market conditions and then reverts to long term (30+ year) market trends.
#

import numpy as np
from numpy.typing import NDArray

def generate_returns(
    n_full: int,
    nsims: int,
    corr_matrix: NDArray[np.float64],
    # Equity Parameters
    initial_equity_mu: float,
    long_term_equity_mu: float,
    initial_equity_sigma: float,
    long_term_equity_sigma: float,
    # Bond Parameters
    initial_bond_mu: float,
    long_term_bond_mu: float,
    initial_bond_sigma: float,
    long_term_bond_sigma: float,
    # Inflation Parameters
    initial_inflation_mu: float,
    long_term_inflation_mu: float,
    initial_inflation_sigma: float,
    long_term_inflation_sigma: float
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Generate Monte Carlo equity, bond, and inflation returns (quarterly) 
    using a mean-reversion model and correlated shocks.

    Args:
        n_full: The number of full years to simulate.
        nsims: The number of Monte Carlo simulations to run.
        corr_matrix: The 3x3 correlation matrix for (Equity, Bond, Inflation).
        
        ... (All required market assumption parameters) ...

    Returns:
        A tuple of (equity_r_q, bond_r_q, infl_q) - all 3D numpy arrays 
        [nsims, n_quarters].
    """
    
    def mr_params(i_mu, lt_mu, i_sig, lt_sig, half_life=10):
        """Calculates mean-reverting mu and sigma for a given time period."""
        revert = np.log(2) / half_life
        mu_t = lt_mu + (i_mu - lt_mu) * np.exp(-revert * np.arange(n_full))
        sig_t = lt_sig + (i_sig - lt_sig) * np.exp(-revert * np.arange(n_full))
        return mu_t, sig_t

    # --- 1. Calculate Annual Mean-Reverting Parameters ---
    eq_mu, eq_sig = mr_params(initial_equity_mu, long_term_equity_mu,
                              initial_equity_sigma, long_term_equity_sigma)
    bo_mu, bo_sig = mr_params(initial_bond_mu, long_term_bond_mu,
                              initial_bond_sigma, long_term_bond_sigma)
    inf_mu, inf_sig = mr_params(initial_inflation_mu, long_term_inflation_mu,
                                initial_inflation_sigma, long_term_inflation_sigma)

    # --- 2. Convert to Quarterly Parameters ---
    eq_mu_q = (1 + eq_mu) ** 0.25 - 1
    eq_sig_q = eq_sig / np.sqrt(4)
    bo_mu_q = (1 + bo_mu) ** 0.25 - 1
    bo_sig_q = bo_sig / np.sqrt(4)
    inf_mu_q = (1 + inf_mu) ** 0.25 - 1
    inf_sig_q = inf_sig / np.sqrt(4)

    n_quarters = n_full * 4
    
    # Repeat annual values four times to get quarterly values
    eq_mu_q = np.repeat(eq_mu_q, 4)
    bo_mu_q = np.repeat(bo_mu_q, 4)
    inf_mu_q = np.repeat(inf_mu_q, 4)
    eq_sig_q = np.repeat(eq_sig_q, 4)
    bo_sig_q = np.repeat(bo_sig_q, 4)
    inf_sig_q = np.repeat(inf_sig_q, 4)
    
    # --- 3. Generate Correlated Quarterly Shocks ---
    L = np.linalg.cholesky(corr_matrix)
    # Generate random shocks: [nsims, n_quarters, 3 assets] @ [3 assets, 3 assets]
    shocks = np.random.randn(nsims, n_quarters, 3) @ L.T

    # --- 4. Calculate Quarterly Returns and Inflation ---
    equity_r_q = eq_mu_q + eq_sig_q * shocks[:, :, 0]
    bond_r_q = bo_mu_q + bo_sig_q * shocks[:, :, 1]
    
    # Inflation shock, capped at -0.01/4 (to prevent excessive deflation)
    infl_q = np.maximum(inf_mu_q + inf_sig_q * shocks[:, :, 2], -0.01/4)
    
    return equity_r_q, bond_r_q, infl_q

def calculate_annual_inflation(
    quarterly_inflation_rates: NDArray, 
    prior_year_inflation_index: float
) -> tuple[float, float]:
    """
    Calculates the annual inflation rate and the new cumulative inflation index 
    for the current year.
    
    Args:
        quarterly_inflation_rates: A 1D array of 4 quarterly inflation rates for the current year.
        prior_year_inflation_index: The cumulative inflation index from the prior year.
        
    Returns:
        tuple[float, float]: (annual_inflation_rate, current_cumulative_index)
    """
    
    # 1. Calculate the annual rate by compounding the four quarterly rates
    # Logic extracted from: np.prod(1 + infl_q[sim_idx,quarter_index:quarter_index+4]) - 1
    annual_infl_r = np.prod(1 + quarterly_inflation_rates) - 1
    
    # 2. Calculate the new cumulative index
    # Logic extracted from: inflation_index[year_idx-1] * (1 + annual_infl_r)
    current_cumulative_index = prior_year_inflation_index * (1 + annual_infl_r)
    
    return annual_infl_r, current_cumulative_index
