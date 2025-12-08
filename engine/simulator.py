# engine.simulator.py

import numpy as np
import pandas as pd
import copy
import math
from typing import List, Dict, Tuple, Any

# --- Utilities and Models ---
from utils.xml_loader import DEFAULT_SETUP, DEFAULT_ACCOUNTS
from models import PlannerInputs

# --- Configuration Imports
from config.expense_assumptions import (
    medicare_start_age,
    medicare_part_b_base_2026,
    medicare_supplement_annual,
    irmaa_brackets_start_year,
    mortgage_payoff_year,
    mortgage_monthly_until_payoff,
    property_tax_and_insurance,
    car_replacement_cycle,
    car_cost_today,
    car_inflation,
    home_repair_prob,
    home_repair_mean,
    home_repair_shape,
    lumpy_expenses
)
from config.market_assumptions import (
    initial_inflation_mu,
    initial_inflation_sigma,
    long_term_inflation_mu,
    long_term_inflation_sigma,
    years_to_revert,
    initial_equity_mu,
    initial_equity_sigma,
    long_term_equity_mu,
    long_term_equity_sigma,
    initial_bond_mu,
    initial_bond_sigma,
    long_term_bond_mu,
    long_term_bond_sigma,
    corr_matrix
)

from engine.rmd_tables import get_rmd_factor
from engine.def457b_tables import get_def457b_factor
from engine.roth_optimizer import optimal_roth_conversion

from engine.accounts_income import AccountsIncomeEngine # importing a class here

from engine.tax_planning import estimate_taxable_gap
from engine.tax_engine import calculate_taxes
from engine.market_generator import generate_returns, calculate_annual_inflation
        
class RetirementSimulator:
    """
    Runs Monte Carlo simulations for retirement planning, calculating taxes 
    and optimizing Roth conversions based on user inputs.
    """
    def __init__(self, inputs: PlannerInputs):

        # -----------------------
        # STEP 1: Initialize Inputs and Core Attributes
        # -----------------------
        self.inputs = inputs

        # Set all input fields as class attributes (e.g., self.nsims, self.current_year, etc.)
        for field, value in inputs.__dict__.items():
            setattr(self, field, value)

        # -----------------------
        # STEP 2: Normalize Accounts and Set Initial Balances
        # -----------------------
        # Must be called first to define 'self.initial_accounts' 
        self._normalize_accounts() 
        
        # Store initial balances for reference (Fix: use self.initial_accounts)
        self.initial_balances = self.initial_accounts 

        # -----------------------
        # STEP 3: Define Simulation Timeframe and Ages
        # -----------------------
        
        # Calculate current ages
        self.current_age_p1 = self.current_year - self.person1_birth_year
        # Use person1's year as fallback if person2_birth_year is None
        self.current_age_p2 = self.current_year - (self.person2_birth_year or self.person1_birth_year)

        # N = The length of the main simulation loop (Year 0 onwards, e.g., age 60 to 89 is 30 years)
        min_current_age = min(self.current_age_p1, self.current_age_p2)
        self.num_years = self.end_age - min_current_age 
        
        # The full path length (N_full) includes two lookback years (Y-2, Y-1)
        self.n_full = self.num_years + 2 

        # Define simulation start parameters
        start_year = self.current_year - 2
        start_age_p1_path = self.current_age_p1 - 2
        start_age_p2_path = self.current_age_p2 - 2

        # Create the full arrays (Length n_full)
        self.years = np.arange(start_year, start_year + self.n_full).tolist()
        self.person1_ages = np.arange(start_age_p1_path, start_age_p1_path + self.n_full).tolist()
        self.person2_ages = np.arange(start_age_p2_path, start_age_p2_path + self.n_full).tolist()

        # Initialize random number generator for stoicastic home repair model
        self.rng = np.random.default_rng()
        
        # -----------------------
        # STEP 4: Initialize Path Storage
        # -----------------------
        # Initialization to None is appropriate since 'run_simulation' calculates the required size
        # and initializes them to numpy arrays of zeros.
        self.portfolio_paths = None
        self.conversion_paths = None
        self.account_paths = None
        self.plan_paths = None
        self.taxes_paths = None
        self.magi_paths = None
        self.base_spending_paths = None
        self.lumpy_spending_paths = None
        self.ssbenefit_paths = None
        self.portfolio_withdrawal_paths = None
        self.trust_income_paths = None
        self.rmd_paths = None
        self.def457b_income_paths = None
        self.pension_paths = None
        self.medicare_paths = None
        self.gifting_paths = None
        self.travel_paths = None

        self.debug_log = []

        # -----------------------
        # STEP 5: Initialize Accounts/Income Engine
        # -----------------------
        # This must happen after self.years, self.nsims, and ages are defined.
        self.accounts_income = AccountsIncomeEngine(
            years=self.years,
            num_sims=self.nsims,
            inputs=self.inputs,
            annual_inflation=None,
            ages_person1=self.person1_ages,
            ages_person2=self.person2_ages,
            return_trajectories=True
        )
        # Link the inputs object to the engine, which is necessary for the engine's methods 
        # (e.g., compute_rmds, compute_pension_income) to access parameters like birth years, etc.
        self.accounts_income.inputs = self.inputs

        #total_years_in_array = self.num_years + 2 # Includes Year -2, Year -1, Year 0 (Current Year), ..., Year N-1
        
        # Initialize storage arrays 
        self.portfolio_paths = None
        self.conversion_paths = None
        self.account_paths = None
        self.plan_paths = None
        self.taxes_paths = None
        self.magi_paths = None
        self.base_spending_paths = None
        self.lumpy_spending_paths = None
        self.ssbenefit_paths = None
        self.portfolio_withdrawal_paths = None
        self.trust_income_paths = None
        self.rmd_paths = None
        self.def457b_income_paths = None
        self.pension_paths = None
        self.medicare_paths = None
        self.gifting_paths = None
        self.travel_paths = None 

        self.debug_log = []

        self.accounts_income = AccountsIncomeEngine(
            years=self.years,
            num_sims=self.nsims,
            inputs=self.inputs,
            annual_inflation=None, 
            ages_person1=self.person1_ages,
            ages_person2=self.person2_ages,
            return_trajectories=True
        )
        self.accounts_income.inputs = self.inputs
    # =========================================================================
    # 1. CORE SIMULATION RUNNER
    # =========================================================================
    def run_simulation(self):
        """Runs the Monte Carlo simulation over all paths."""
        num_paths = self.nsims
        num_years = self.num_years
        
        # Initialize storage arrays (restored based on original snippet)
        self.portfolio_paths = np.zeros((num_paths, num_years))
        self.conversion_paths = np.zeros((num_paths, num_years))
        self.taxes_paths = np.zeros((num_paths, num_years))
        self.magi_paths = np.zeros((num_paths, num_years))
        self.base_spending_paths = np.zeros((num_paths, num_years))
        self.lumpy_spending_paths = np.zeros((num_paths, num_years))
        self.ssbenefit_paths = np.zeros((num_paths, num_years))
        self.portfolio_withdrawal_paths = np.zeros((num_paths, num_years))
        self.trust_income_paths = np.zeros((num_paths, num_years))
        self.rmd_paths = np.zeros((num_paths, num_years))
        self.def457b_income_paths = np.zeros((num_paths, num_years))
        self.pension_paths = np.zeros((num_paths, num_years))
        self.medicare_paths = np.zeros((num_paths, num_years))
        self.gifting_paths = np.zeros((num_paths, num_years))
        self.travel_paths = np.zeros((num_paths, num_years))

        # Account paths stores the *final* account balance structure for each path/year
        self.account_paths = [[None] * num_years for _ in range(num_paths)] 
        
        # The remaining arrays (e.g., plan_paths, base_spending_paths) should be initialized here 
        # if they are guaranteed to be returned by _summarize_results

        for path_index in range(num_paths):
            self._run_one_path(path_index)

        return self._summarize_results()


    # =========================================================================
    # 2. SINGLE PATH LOGIC
    # =========================================================================
    def _run_one_path(self, path_index: int):
        """Runs a single simulation path."""
        
        print(f">>> STARTING PATH {path_index} <<<")
        
        # Initialize state for this path
        current_accounts = copy.deepcopy(self.initial_accounts)

        annual_rmd_required = 0.0   # This year's total RMD (calculated once, drawn quarterly)
        
        # Initialize path-specific history arrays
        # MAGI history needs to be long enough to track 2 years prior for IRMAA (index offset)
        magi_path = [0.0] * 2 + [0.0] * self.num_years
        
        equity_q_path, bond_q_path = self._generate_market_path()

        inflation_path = self._generate_inflation_path()
        inflation_path = self._generate_inflation_path()
        
        for i in range(self.num_years):
            year_index = i
            current_year = self.current_year + i
            current_inflation_index = inflation_path[i]
            # Update ages
            current_age_person1 = self.current_age_p1 + i
            current_age_person2 = self.current_age_p2 + i if self.current_age_p2 is not None else None
            # =========================================================================
            # --- STEP 1: CALCULATE ANNUAL INCOME AND RMDs ---
            # =========================================================================
            # AGI is calculated progressively
            AGI = 0.0
             
            # RMDs (Required Minimum Distributions)
            annual_rmd_required = self.accounts_income.compute_rmds(current_accounts, current_year)
            self.rmd_paths[path_index, i] = annual_rmd_required
            AGI += annual_rmd_required
            
            # Pension/Def457b Income
            pension_income = self.accounts_income.compute_pension_income(current_year, current_inflation_index)
            def457b_income = self.accounts_income.compute_def457b_income(current_accounts, current_year)
            AGI += pension_income + def457b_income
            self.pension_paths[path_index, i] = pension_income
            self.def457b_income_paths[path_index, i] = def457b_income
            
            # Social Security Income
            ss_benefit = self.accounts_income.compute_ss_benefit(current_year, current_inflation_index)
            ss_taxable = self.accounts_income.compute_taxable_ss(ss_benefit, AGI, self.inputs.filing_status) 
            AGI += ss_taxable # Taxable portion of SS contributes to AGI
            self.ssbenefit_paths[path_index, i] = ss_benefit # Save full benefit amount
            # =========================================================================
            # --- STEP 2: ROTH CONVERSION OPTIMIZATION (Ground Truth Logic) ---
            # =========================================================================
            # We must process conversions sequentially (e.g., Person 2 then Person 1)
            # because the first conversion increases the AGI for the second person.
            
            total_conversion_this_year = 0.0
            
            # --- Person 2 Conversion ---
            p2_trad_accts = [k for k, v in current_accounts.items() 
                             if v.get("owner") == "person2" and v.get("tax") == "traditional"]
            p2_trad_bal = sum(current_accounts[k]["balance"] for k in p2_trad_accts)
            
            conv_p2 = 0.0
            if p2_trad_bal > 0:
                conv_p2 = optimal_roth_conversion(
                    year=current_year,
                    inflation_index=current_inflation_index,
                    filing_status=self.filing_status,
                    AGI_base=AGI, # Current AGI
                    traditional_balance=p2_trad_bal,
                    tax_strategy=self.tax_strategy,
                    irmaa_strategy=self.irmaa_strategy
                )
                
            # Execute Person 2 Conversion (Pro-rata across their traditional accounts)
            if conv_p2 > 0:
                # Find destination Roth (use first found or create new)
                p2_roth_targets = [k for k, v in current_accounts.items() 
                                   if v.get("owner") == "person2" and v.get("tax") == "roth"]
                target_roth = p2_roth_targets[0] if p2_roth_targets else None
                
                # If no Roth exists for Person 2, we technically can't convert 
                # (or we'd need to spawn a new account). For now, skip if no target.
                if target_roth:
                    remaining_to_convert = conv_p2
                    for name in p2_trad_accts:
                        if remaining_to_convert <= 0: break
                        bal = current_accounts[name]["balance"]
                        if bal <= 0: continue
                        
                        # Pro-rata-ish: drain accounts in order until amount met
                        amount = min(bal, remaining_to_convert)
                        current_accounts[name]["balance"] -= amount
                        current_accounts[target_roth]["balance"] += amount
                        remaining_to_convert -= amount
                    
                    # Update AGI for Person 1's calculation
                    AGI += conv_p2
                    total_conversion_this_year += conv_p2

            # --- Person 1 Conversion ---
            p1_trad_accts = [k for k, v in current_accounts.items() 
                             if v.get("owner") == "person1" and v.get("tax") == "traditional"]
            p1_trad_bal = sum(current_accounts[k]["balance"] for k in p1_trad_accts)
            
            conv_p1 = 0.0
            if p1_trad_bal > 0:
                conv_p1 = optimal_roth_conversion(
                    year=current_year,
                    inflation_index=current_inflation_index,
                    filing_status=self.filing_status,
                    AGI_base=AGI, # AGI is now higher due to P2's conversion
                    traditional_balance=p1_trad_bal,
                    tax_strategy=self.tax_strategy,
                    irmaa_strategy=self.irmaa_strategy
                )

            # Execute Person 1 Conversion
            if conv_p1 > 0:
                p1_roth_targets = [k for k, v in current_accounts.items() 
                                   if v.get("owner") == "person1" and v.get("tax") == "roth"]
                target_roth = p1_roth_targets[0] if p1_roth_targets else None
                
                if target_roth:
                    remaining_to_convert = conv_p1
                    for name in p1_trad_accts:
                        if remaining_to_convert <= 0: break
                        bal = current_accounts[name]["balance"]
                        if bal <= 0: continue
                        
                        amount = min(bal, remaining_to_convert)
                        current_accounts[name]["balance"] -= amount
                        current_accounts[target_roth]["balance"] += amount
                        remaining_to_convert -= amount
                    
                    AGI += conv_p1
                    total_conversion_this_year += conv_p1

            self.conversion_paths[path_index, i] = total_conversion_this_year
            # =========================================================================
            # --- STEP 3: TAX CALCULATION ---
            # =========================================================================
            MAGI = AGI # Simplified: MAGI is often close to AGI, conversion is included.
            
            # Get MAGI from 2 years prior for IRMAA calculation (offset by 2)
            magi_two_years_ago = magi_path[year_index] 
            
            total_taxes, federal_tax, medicare_irmaa = calculate_taxes(
                year=current_year,
                inflation_index=current_inflation_index,
                filing_status=self.filing_status,           
                state_of_residence=self.state_of_residence, 
                age1=current_age_person1,
                age2=current_age_person2,
                magi_two_years_ago=magi_two_years_ago,
                AGI=AGI,
                taxable_ordinary=AGI, # Simplified: Assumes AGI is the ordinary income base
                lt_cap_gains=0.0,     # Placeholder for Taxable account gains
                qualified_dividends=0.0, 
                social_security_income=ss_benefit,
            )
# ... after Step 3: TAX CALCULATION ...
            self.taxes_paths[path_index, i] = total_taxes
            self.medicare_paths[path_index, i] = medicare_irmaa
            magi_path[year_index + 2] = MAGI 

            # =========================================================================
            # --- STEP 4: DETERMINE ANNUAL WITHDRAWAL NEED ---
            # =========================================================================
            # This calculates the total cash DESIRED/REQUIRED from the portfolio for the year.

            (total_fixed_expenses, 
             annual_base_spending, 
             annual_lumpy_needs, 
             annual_home_repair,
             annual_travel_desired, 
             annual_gifting_desired) = self._calculate_annual_spending_needs(
                current_year, 
                current_inflation_index
            )
            
            # Save expense paths (Store actual/desired amounts as determined by the planner)
            # NOTE: If your optimizer cuts travel/gifting, you must update these paths in STEP 6.
            self.base_spending_paths[path_index, i] = annual_base_spending
            self.lumpy_spending_paths[path_index, i] = annual_lumpy_needs
            self.travel_paths[path_index, i] = annual_travel_desired
            self.gifting_paths[path_index, i] = annual_gifting_desired
            # You may also need a home_repair_paths

            # Total desired cash for the year (Fixed Expenses + Desired Adjustable + Taxes)
            total_annual_withdrawal_needed = (
                total_fixed_expenses + 
                annual_travel_desired + 
                annual_gifting_desired + 
                total_taxes
            )

            # Cash from non-portfolio sources (Pension, SS, Def457b)
            total_annual_cash_in = (
                pension_income + 
                def457b_income + 
                ss_benefit
            )

            # The net amount to be pulled from the portfolio to cover the gap
            annual_portfolio_draw_needed = max(0.0, total_annual_withdrawal_needed - total_annual_cash_in)
            self.portfolio_withdrawal_paths[path_index, i] = annual_portfolio_draw_needed
            
            # =========================================================================
            # --- STEP 5: QUARTERLY WITHDRAWAL AND INVESTMENT RETURNS ---
            # =========================================================================

            # Annual amounts are divided into four equal quarterly amounts
            quarterly_rmd_draw = annual_rmd_required / 4.0
            quarterly_portfolio_draw_needed = annual_portfolio_draw_needed / 4.0
            
            # Get the four quarterly returns for the current year
            eq_q_returns = equity_q_path[i*4 : (i+1)*4]
            bond_q_returns = bond_q_path[i*4 : (i+1)*4]

            # Track actual tax character of portfolio withdrawals for final MAGI/tax adjustment
            final_ordinary_income_actual = 0.0
            final_ltcg_income_actual = 0.0
            
            # Hardcoded withdrawal order (from withdrawal_engine.py snippet)
            withdrawal_order = ["taxable", "trust", "inherited", "traditional", "roth"] 

            for q in range(4): 
                
                # Total cash needed from portfolio draw (includes the RMD)
                quarterly_total_draw = quarterly_portfolio_draw_needed + quarterly_rmd_draw
                
                # Execute the withdrawal, modifying current_accounts in place
                # Assumes self.accounts_income is an instance of a class that contains _withdraw_from_hierarchy
                withdrawal_result = self.accounts_income._withdraw_from_hierarchy(
                    cash_needed=quarterly_total_draw, 
                    accounts_bal=current_accounts, 
                    order=withdrawal_order,
                )

                final_ordinary_income_actual += withdrawal_result.get("ordinary_inc", 0.0)
                final_ltcg_income_actual += withdrawal_result.get("ltcg_inc", 0.0)
                
                # Apply Quarterly Investment Returns (using separate asset classes)
                market_return_eq = eq_q_returns[q] 
                market_return_bond = bond_q_returns[q]

                for name in current_accounts.keys():
                    acct = current_accounts[name]
                    
                    # Portfolio Blending (Assumes a 70/30 allocation if not specified in account)
                    # NOTE: This must be properly configured from your inputs
                    equity_pct = acct.get('equity_pct', 0.7) 
                    bond_pct = 1.0 - equity_pct
                    
                    # Calculate blended return for the account
                    blended_q_return = (
                        equity_pct * market_return_eq + 
                        bond_pct * market_return_bond
                    )

                    # Apply return to remaining balance
                    acct['balance'] *= (1 + blended_q_return)
                    
                    # Basis tracking: Apply return to basis for taxable accounts
                    if acct.get('tax') == 'taxable':
                        # Basis grows with the market return for taxable accounts
                        basis_growth = acct.get('basis', 0.0) * blended_q_return
                        acct['basis'] = acct.get('basis', 0.0) + basis_growth

            # =========================================================================
            # --- STEP 6: POST-QUARTERLY CLEANUP AND SAVE DATA ---
            # =========================================================================
            
           # Final Portfolio Balance
            self.portfolio_paths[path_index, i] = self._get_total_portfolio(current_accounts)
            
            # Save the final account state for the year
            self.account_paths[path_index][i] = copy.deepcopy(current_accounts)

            # NOTE: Final adjustment of MAGI and taxes based on actual withdrawals (ordinary_income_actual)
            # and re-running the Roth conversion logic may occur here or in Step 2.
            # Assuming the loop continues to the next year (i+1).


    # =========================================================================
    # 3. UTILITY FUNCTIONS — Modular Wiring Only
    # =========================================================================

    def _normalize_accounts(self):
        """Ensure account dicts have required fields."""
        self.initial_accounts = copy.deepcopy(self.inputs.accounts)
        for acct in self.initial_accounts.values():
            acct.setdefault("balance", 0.0)
            acct.setdefault("basis", 0.0)
            acct.setdefault("tax", "traditional")
            acct.setdefault("owner", "person1")
            acct.setdefault("mandatory_yield", 0.0)
            acct.setdefault("ordinary_pct", 0.1)

    def _get_total_portfolio(self, accounts: Dict) -> float:
        return sum(acct.get("balance", 0.0) for acct in accounts.values())

    def _generate_market_path(self) -> Tuple[List[float], List[float]]:
        """
        One path of *quarterly* portfolio returns using market_generator.
        Returns separate lists for Equity and Bond quarterly returns.
        """
        from config.market_assumptions import (
            corr_matrix, initial_equity_mu, long_term_equity_mu,
            initial_equity_sigma, long_term_equity_sigma, initial_bond_mu, 
            long_term_bond_mu, initial_bond_sigma, long_term_bond_sigma, 
            initial_inflation_mu, long_term_inflation_mu, 
            initial_inflation_sigma, long_term_inflation_sigma
        )
        
        # NOTE: generate_returns returns three arrays (Equity, Bond, Inflation)
        eq_q, bond_q, _ = generate_returns( 
            n_full=self.num_years, 
            nsims=1,
            corr_matrix=corr_matrix,
            initial_equity_mu=initial_equity_mu,
            long_term_equity_mu=long_term_equity_mu,
            initial_equity_sigma=initial_equity_sigma,
            long_term_equity_sigma=long_term_equity_sigma,
            initial_bond_mu=initial_bond_mu,
            long_term_bond_mu=long_term_bond_mu,
            initial_bond_sigma=initial_bond_sigma,
            long_term_bond_sigma=long_term_bond_sigma,
            initial_inflation_mu=initial_inflation_mu,
            long_term_inflation_mu=long_term_inflation_mu,
            initial_inflation_sigma=initial_inflation_sigma,
            long_term_inflation_sigma=long_term_inflation_sigma,
        )
        
        # Explicitly return a tuple of two lists, ensuring two items are unpacked.
        # [0, :] flattens the (1, N) array into a 1D array before converting to a list.
        return (eq_q[0, :].tolist(), bond_q[0, :].tolist())
    
    def _generate_inflation_path(self) -> List[float]:
        """One path of cumulative inflation index."""
        _, _, infl_q = generate_returns(
            n_full=self.num_years,
            nsims=1,
            corr_matrix=corr_matrix,
            initial_equity_mu=initial_equity_mu,
            long_term_equity_mu=long_term_equity_mu,
            initial_equity_sigma=initial_equity_sigma,
            long_term_equity_sigma=long_term_equity_sigma,
            initial_bond_mu=initial_bond_mu,
            long_term_bond_mu=long_term_bond_mu,
            initial_bond_sigma=initial_bond_sigma,
            long_term_bond_sigma=long_term_bond_sigma,
            initial_inflation_mu=initial_inflation_mu,
            long_term_inflation_mu=long_term_inflation_mu,
            initial_inflation_sigma=initial_inflation_sigma,
            long_term_inflation_sigma=long_term_inflation_sigma,
        )
        cumulative = [1.0]
        for y in range(self.num_years):
            q_rates = infl_q[0, y*4:(y+1)*4]
            ann_rate, _ = calculate_annual_inflation(q_rates, cumulative[-1])
            cumulative.append(cumulative[-1] * (1 + ann_rate))
        return cumulative[1:]  # drop year 0 = 1.0

    def _get_social_security_benefit(self, current_year: int, inflation_index: float) -> float:
        return calculate_ss_benefit(
            ages_person1=np.array([self.current_age_p1 + i for i in range(self.num_years)]),
            ages_person2=np.array([self.current_age_p2 + i for i in range(self.num_years)]),
            person1_ss_age_years=self.person1_ss_age_years,
            person1_ss_age_months=self.person1_ss_age_months,
            person1_birth_month=self.person1_birth_month,
            person1_ss_fra=self.person1_ss_fra,
            person1_birth_year=self.person1_birth_year,
            person2_ss_age_years=self.person2_ss_age_years,
            person2_ss_age_months=self.person2_ss_age_months,
            person2_birth_month=self.person2_birth_month,
            person2_ss_fra=self.person2_ss_fra,
            person2_birth_year=self.person2_birth_year,
            ss_fail_year=self.ss_fail_year,
            ss_fail_percent=self.ss_fail_percent,
            inflation_index=np.array([inflation_index]),
            year=current_year,
            year_idx=current_year - self.current_year,
            get_ss_multiplier_func=self._get_ss_multiplier,  # you already have this somewhere
        )

    def _calculate_taxable_ss(self, ss_benefit: float, AGI: float) -> Tuple[float, float]:
        taxable = calculate_taxable_ss_portion(ss_benefit)
        return taxable, ss_benefit - taxable

    def _calculate_rmds(self, accounts: Dict, age1: int, age2: int, year: int) -> float:
        rmd_total = 0.0
        for name, acct in accounts.items():
            if acct.get("tax") not in ["traditional", "inherited"]:
                continue
            bal = acct.get("balance", 0.0)
            owner_age = age1 if acct.get("owner") == "person1" else age2
            factor = get_rmd_factor(owner_age, year)
            if factor > 0:
                rmd = bal / factor
                acct["balance"] -= rmd
                rmd_total += rmd
        return rmd_total

    def _get_pension_benefit(self, current_year: int, inflation_index: float) -> float:
        return calculate_pension_income(
            ages_person1=np.array([self.current_age_p1 + i for i in range(self.num_years)]),
            ages_person2=np.array([self.current_age_p2 + i for i in range(self.num_years)]),
            person1_pension_age_years=self.person1_pension_age_years,
            person1_birth_month=self.person1_birth_month,
            person1_pension_age_months=self.person1_pension_age_months,
            person1_pension_amount=self.person1_pension_amount,
            person2_pension_age_years=self.person2_pension_age_years,
            person2_birth_month=self.person2_birth_month,
            person2_pension_age_months=self.person2_pension_age_months,
            person2_pension_amount=self.person2_pension_amount,
            year_idx=current_year - self.current_year,
        )

    def _get_def457b_income(self, current_year: int, inflation_index: float) -> float:
        _, def457b_draw, _ = calculate_mandatory_account_draws(
            accounts_bal={k: v.get("balance", 0.0) for k, v in self.initial_accounts.items()},
            accounts_def=self.initial_accounts,
            year=current_year,
            birth_years={"person1": self.person1_birth_year, "person2": self.person2_birth_year},
            get_def457b_factor_func=get_def457b_factor,
        )
        return def457b_draw

    def _get_cash_needed(self, current_year: int, inflation_index: float) -> float:
        base = self.base_annual_spending * inflation_index
        # Add travel, gifting, lumpy, etc. here when ready
        return base + self.travel + self.gifting

    def _get_withdrawal_order(self, year: int) -> List[str]:
        return ["taxable", "trust", "inherited", "traditional", "roth"]

    def _execute_withdrawals(self, accounts: Dict, amount: float, order: List[str]) -> Tuple[float, float]:
        engine = AccountsIncomeEngine(...)  # or inject via __init__ if preferred
        result = engine._withdraw_from_hierarchy(amount, accounts, order)
        return result["ordinary_inc"], result["ltcg_inc"]

    def _apply_market_returns(self, accounts: Dict, market_return: float):
        for acct in accounts.values():
            if "balance" in acct:
                acct["balance"] *= (1 + market_return)

    def _calculate_annual_spending_needs(self, 
        current_year: int, 
        inflation_index: float) -> Tuple[float, float, float, float, float, float]:
        """
        Calculates total annual expenses (fixed and adjustable).
        
        Returns: 
            (total_fixed_expenses, annual_base_spending, 
             annual_lumpy_needs, annual_home_repair, 
             annual_travel_desired, annual_gifting_desired)
        """
        
        from config.expense_assumptions import (
            mortgage_payoff_year, mortgage_monthly_until_payoff,
            property_tax_and_insurance, car_replacement_cycle,
            car_cost_today, lumpy_expenses,
            home_repair_prob, home_repair_mean, home_repair_shape,
        )
        
        # 1. Base Spending (Inflation-Adjusted)
        annual_base_spending = self.base_annual_spending * inflation_index
        
        # 2. Housing/Fixed Expenses (Mortgage is fixed, P&T is inflation-adjusted)
        mortgage_expense = 0.0
        if current_year <= mortgage_payoff_year:
            mortgage_expense = mortgage_monthly_until_payoff 
            
        property_and_tax = property_tax_and_insurance * inflation_index 

        # 3. Lumpy Expenses (Check against configuration list)
        annual_lumpy_needs = 0.0
        for item in lumpy_expenses:
            if item.get("year") == current_year:
                annual_lumpy_needs += item.get("amount", 0.0) * inflation_index 
                
        # 4. Car Replacement (Every 'car_replacement_cycle' years, inflated)
        car_expense = 0.0
        years_since_start = current_year - self.current_year
        if years_since_start % car_replacement_cycle == 0 and years_since_start >= 0:
            car_expense = car_cost_today * inflation_index
            
        # 5. Stochastic Home Repair (Restored Logic)
        annual_home_repair = 0.0
        if self.rng.random() < home_repair_prob:
            # Draw from log-normal distribution (requires self.rng to be a seeded generator)
            mu_log = np.log(home_repair_mean) - (home_repair_shape ** 2) / 2
            annual_home_repair = self.rng.lognormal(mu_log, home_repair_shape) * inflation_index
        
        # TOTAL FIXED EXPENSES (Required draw from portfolio/cash flow)
        total_fixed_expenses = (
            annual_base_spending +
            mortgage_expense +
            property_and_tax +
            annual_lumpy_needs +
            car_expense +
            annual_home_repair
        )
        
        # 6. Discretionary Spending (Dynamically Adjustable to manage tax/IRMAA)
        # These are the DESIRED amounts, which may be cut by the planner (Step 2/3)
        annual_travel_desired = self.travel * inflation_index
        annual_gifting_desired = self.gifting * inflation_index
        
        return (
            total_fixed_expenses, 
            annual_base_spending, 
            annual_lumpy_needs, 
            annual_home_repair,
            annual_travel_desired, 
            annual_gifting_desired
        )

    # =========================================================================
    # 4. RESULTS SUMMARIZER
    # =========================================================================
    def _summarize_results(self) -> Dict[str, Any]:
        """
        Summarizes the final state of all simulation paths, performing required 
        data transposition and ensuring all paths are defensive arrays.
        """
        import numpy as np
        
        # If simulation paths were not successfully generated, return defensive empty results
        if self.portfolio_paths is None or self.portfolio_paths.size == 0:
             # Defensive returns for all paths to prevent AxisError in plotting
             return {
                "success_rate": 0.0,
                "avoid_ruin_rate": 0.0,
                "median_final": 0.0,
                "p10_final": 0.0,
                "portfolio_paths": np.array([]),
                "account_paths": {},
                "conversion_paths": np.array([]),
                "taxes_paths": np.array([]),
                "magi_paths": np.array([]),
                "trust_income_paths": np.array([]),
                "ssbenefit_paths": np.array([]),
                "portfolio_withdrawal_paths": np.array([]),
                "rmd_paths": np.array([]),
                "def457b_income_paths": np.array([]),
                "pension_paths": np.array([]),
                "medicare_paths": np.array([]),
                "travel_paths": np.array([]),
                "gifting_paths": np.array([]),
                "base_spending_paths": np.array([]),
                "lumpy_spending_paths": np.array([]),
                "plan_paths": np.array([]),
             }

        success = self.inputs.success_threshold
        avoid_ruin = self.inputs.avoid_ruin_threshold
        portfolio_end = self.portfolio_paths[:, -1]
        
        success_rate = np.mean(portfolio_end > success) * 100
        minimum_annual_balance = np.min(self.portfolio_paths, axis=1)
        avoid_ruin_rate = np.mean(minimum_annual_balance > avoid_ruin) * 100
        
        # --- START FIX: Transpose account_paths and filter to XML-defined names ---
        num_paths, num_years = self.portfolio_paths.shape

        account_paths_list = getattr(self, 'account_paths', [])
        
        # The set of valid names are those defined in the user's XML inputs.
        xml_account_names = set(self.inputs.accounts.keys())

        # 1. Determine all unique account names *actually* present in the results
        account_names_in_results = set()
        for path in account_paths_list:
            for account_state in path:
                if account_state:
                    account_names_in_results.update(account_state.keys())

        # Filter: Only transpose accounts that were defined in the XML (e.g., keep 'Roth_IRA', drop 'Roth')
        names_to_transpose = xml_account_names.intersection(account_names_in_results)

        # 2. Initialize the transposed structure
        transposed_account_paths = {}
        for name in names_to_transpose:
            transposed_account_paths[name] = np.zeros((num_paths, num_years))

        # 3. Fill the transposed structure
        for i in range(num_paths):
            for j in range(num_years):
                account_state = account_paths_list[i][j]
                if account_state is not None:
                    for name in names_to_transpose:
                        # Extract the balance, defaulting to 0.0
                        # Balance is nested: acct_state['AccountName']['balance']
                        balance = account_state.get(name, {}).get('balance', 0.0)
                        transposed_account_paths[name][i, j] = balance
        # --- END Transposition and Filtering FIX ---

        result = {
            "success_rate": success_rate,
            "avoid_ruin_rate": avoid_ruin_rate,
            "median_final": np.median(portfolio_end),
            "p10_final": np.percentile(portfolio_end, 10),
            
            # --- PATHS ---
            "portfolio_paths": self.portfolio_paths, 
            "account_paths": transposed_account_paths, # <-- THE CORRECTED, TRANSPOSED, AND FILTERED DICTIONARY
            
            # All other paths are retrieved and defaulted to an empty array to prevent AxisError
            "conversion_paths": getattr(self, 'conversion_paths', np.array([])),
            "taxes_paths": getattr(self, 'taxes_paths', np.array([])),
            "magi_paths": getattr(self, 'magi_paths', np.array([])),
            "trust_income_paths": getattr(self, 'trust_income_paths', np.array([])),
            "ssbenefit_paths": getattr(self, 'ssbenefit_paths', np.array([])),
            "portfolio_withdrawal_paths": getattr(self, 'portfolio_withdrawal_paths', np.array([])),
            "rmd_paths": getattr(self, 'rmd_paths', np.array([])),
            "def457b_income_paths": getattr(self, 'def457b_income_paths', np.array([])),
            "pension_paths": getattr(self, 'pension_paths', np.array([])),
            "medicare_paths": getattr(self, 'medicare_paths', np.array([])),
            "travel_paths": getattr(self, 'travel_paths', np.array([])),
            "gifting_paths": getattr(self, 'gifting_paths', np.array([])),
            "base_spending_paths": getattr(self, 'base_spending_paths', np.array([])),
            "lumpy_spending_paths": getattr(self, 'lumpy_spending_paths', np.array([])),
            "plan_paths": getattr(self, 'plan_paths', np.array([])),
        }
        return result
