# engine.simulator.py

import numpy as np
import pandas as pd
import copy
import math
from typing import List, Dict, Tuple, Any

# --- Utilities and Models ---
from utils.xml_loader import DEFAULT_SETUP, DEFAULT_ACCOUNTS
from utils.currency import clean_percent
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
from engine.roth_optimizer import optimal_roth_conversion

from engine.accounts_income import AccountsIncomeEngine # importing a class here
from engine.withdrawal_engine import WithdrawalEngine # importing a class here

from engine.tax_engine import calculate_taxes, get_effective_marginal_rates
from engine.tax_planning import get_tax_planning_targets
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
        self.current_age_p1 = self.current_year - self.person1_birth_year - 1 # age at start of year
        # Use person1's year as fallback if person2_birth_year is None
        self.current_age_p2 = self.current_year - (self.person2_birth_year or self.person1_birth_year) - 1 # age at start of year

        # N = The length of the main simulation loop
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
        self.plan_paths = None
        self.taxes_paths = None
        self.magi_paths = None
        self.base_spending_paths = None
        self.mortgage_expense_paths = None
        self.lumpy_spending_paths = None
        self.ssbenefit_paths = None
        self.portfolio_withdrawal_paths = None
        self.trust_income_paths = None
        self.rmd_paths = None
        self.salary_paths = None
        self.def457b_income_paths = None
        self.pension_paths = None
        self.medicare_paths = None
        self.gifting_paths = None
        self.travel_paths = None

        self.account_paths = None

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
        

        # -----------------------
        # STEP 6: Instantiate a WithdrawalEngine in the simulator
        # -----------------------
        self.withdrawal_engine = WithdrawalEngine(
            inputs=self.inputs,
            accounts_metadata=self.initial_accounts  # metadata for account types, basis, ordinary_pct, etc.
        )

    # =========================================================================
    # 1. CORE SIMULATION RUNNER
    # =========================================================================
    def run_simulation(self):
        """Runs the Monte Carlo simulation over all paths."""
        num_paths = self.nsims
        num_years = self.num_years
        
        # Initialize storage arrays
        # n_full includes 2 prior years to initialize MAGI, portfolio data
        self.portfolio_paths = np.zeros((num_paths, self.n_full))
        self.conversion_paths = np.zeros((num_paths, self.n_full))
        self.plan_paths = np.zeros((num_paths, self.n_full))
        self.taxes_paths = np.zeros((num_paths, self.n_full))
        self.magi_paths = np.zeros((num_paths, self.n_full))
        self.base_spending_paths = np.zeros((num_paths, self.n_full))
        self.mortgage_expense_paths = np.zeros((num_paths, self.n_full))
        self.lumpy_spending_paths = np.zeros((num_paths, self.n_full))
        self.ssbenefit_paths = np.zeros((num_paths, self.n_full))
        self.portfolio_withdrawal_paths = np.zeros((num_paths, self.n_full))
        self.trust_income_paths = np.zeros((num_paths, self.n_full))
        self.rmd_paths = np.zeros((num_paths, self.n_full))
        self.salary_paths = np.zeros((num_paths, self.n_full))
        self.def457b_income_paths = np.zeros((num_paths, self.n_full))
        self.pension_paths = np.zeros((num_paths, self.n_full))
        self.medicare_paths = np.zeros((num_paths, self.n_full))
        self.gifting_paths = np.zeros((num_paths, self.n_full))
        self.travel_paths = np.zeros((num_paths, self.n_full))

        # Account paths stores the *final* account balance structure for each path/year
        self.account_paths = [[None] * self.n_full for _ in range(num_paths)] 
        
        for path_index in range(num_paths):
            self._run_one_path(path_index)

        return self._summarize_results()


    # =========================================================================
    # 2. SINGLE PATH LOGIC
    # =========================================================================
    def _run_one_path(self, path_index: int):
        """Runs a single simulation path."""
        
        if path_index % 100 == 0:
            print(f">>> STARTING PATH {path_index} <<<")
        
        # Initialize state for this path
        current_accounts = copy.deepcopy(self.initial_accounts)

        # Initialize path-specific history arrays
        # MAGI history needs to be long enough to track 2 years prior for IRMAA (index offset)
        magi_path = [0.0] * 2 + [0.0] * self.num_years
        
        equity_q_path, bond_q_path = self._generate_market_path()

        inflation_path = self._generate_inflation_path()
        
        # --- PRE-SIMULATION SEED ---
        # Index 0 -> 2024 (initial_accounts)
        self.portfolio_paths[path_index, 0] = self._get_total_portfolio(self.initial_accounts)
        self.account_paths[path_index][0] = copy.deepcopy(self.initial_accounts)
        self.magi_paths[path_index, 0] = self.inputs.magi_1  # MAGI1
        # Index 1 -> 2025 (input accounts)
        self.portfolio_paths[path_index, 1] = self._get_total_portfolio(self.initial_accounts)
        self.account_paths[path_index][1] = copy.deepcopy(self.initial_accounts)
        self.magi_paths[path_index, 1] = self.inputs.magi_2  # MAGI2
        
        for i in range(self.num_years):
            year_index = i + 2 # start simulations at index 2
            current_accounts = copy.deepcopy(self.account_paths[path_index][year_index - 1])
            current_year = self.current_year + i
            current_inflation_index = inflation_path[i]
            
            #print(f"Year {current_year} inflation index {current_inflation_index:.2f}")
            
            # Update ages
            current_age_person1 = self.current_age_p1 + i
            current_age_person2 = self.current_age_p2 + i if self.current_age_p2 is not None else None
            
            # initialize variables for this year of simulation
            annual_rmd_required = 0.0
            final_ordinary_income_actual = 0.0
            final_ltcg_income_actual = 0.0
            total_taxes_final = 0.0
            spend_plan = 0.0
            # AGI_estimate is calculated progressively
            AGI_estimate = 0.0
            ORD_estimate = 0.0
            LTCG_estimate = 0.0
            actual_withdrawals = 0.0
        
            # =========================================================================
            # --- STEP 1: CALCULATE ANNUAL INCOME AND RMDs ---
            # =========================================================================

            # RMDs (Required Minimum Distributions)
            # Unpack the total RMD for AGI_estimate and the dictionary for quarterly withdrawals.
            annual_rmd_required, rmds_by_account_annual = self.accounts_income.compute_rmds(current_year, current_accounts)
            
            # ORDINARY Income streams
            salary_income = self.accounts_income.compute_salary_income(current_year, current_inflation_index)
            pension_income = self.accounts_income.compute_pension_income(current_year, current_inflation_index)
            def457b_income = self.accounts_income.compute_def457b_income(current_year, current_inflation_index, current_accounts)
            trust_income = self.accounts_income.compute_trust_income(current_year, current_inflation_index, current_accounts)
            
            # Social Security Income
            ss_benefit = self.accounts_income.compute_ss_benefit(current_year, current_inflation_index)
            
            # Add all fully taxable (Ordinary) income sources to AGI_estimate
            AGI_estimate += salary_income + pension_income + def457b_income + trust_income
            ORD_estimate += salary_income + pension_income + def457b_income + trust_income

            # Add salary and pension income to final ordinary income
            # For consistency 457b and Trust income added in quartly loop when actual account draws are taken
            final_ordinary_income_actual += salary_income + pension_income
            
            # Save paths for tracking/reporting
            self.rmd_paths[path_index, year_index] = annual_rmd_required
            self.salary_paths[path_index, year_index] = salary_income
            self.pension_paths[path_index, year_index] = pension_income
            self.def457b_income_paths[path_index, year_index] = def457b_income
            self.trust_income_paths[path_index, year_index] = trust_income
            self.ssbenefit_paths[path_index, year_index] = ss_benefit
 
            # =========================================================================
            # --- STEP 2: Initial Estimated TAX CALCULATION ---
            # --- ordinary income only
            # =========================================================================
            # Get MAGI from 2 years prior for IRMAA calculation (offset by 2)
            magi_two_years_ago = self.magi_paths[path_index, year_index - 2] 
            
            state_tax, federal_tax, medicare_irmaa = calculate_taxes(
                year=current_year,
                inflation_index=current_inflation_index,
                filing_status=self.filing_status,           
                state_of_residence=self.state_of_residence, 
                age1=current_age_person1,
                age2=current_age_person2,
                magi_two_years_ago=magi_two_years_ago,
                AGI=AGI_estimate,
                taxable_ordinary=ORD_estimate, 
                lt_cap_gains=0.0,              
                qualified_dividends=0.0, 
                social_security_income=ss_benefit,
            )
            initial_estimated_tax = state_tax + federal_tax

            #print(f"STEP 2: Year {current_year}  AGI Estimate Ord. Income Only=${AGI_estimate:,.0f}  Inital Tax Estimates: State=${state_tax:,.0f} Federal=${federal_tax:,.0f}")

            # =========================================================================
            # --- STEP 3: DETERMINE ANNUAL WITHDRAWAL NEED ---
            # --- Include full desired travel and gifting for "maximal" estimate
            # =========================================================================
            # This calculates the total cash DESIRED from the portfolio for the year.
            (annual_base_spending,
             mortgage_expense,
             lumpy_needs,
             annual_travel_desired,
             annual_gifting_desired) = self._calculate_annual_spending_needs(
                current_year, 
                current_inflation_index
            )

            self.base_spending_paths[path_index, year_index] = annual_base_spending
            self.mortgage_expense_paths[path_index, year_index] = mortgage_expense
            self.lumpy_spending_paths[path_index, year_index] = lumpy_needs

            initial_expense_estimate = (
                annual_base_spending +
                mortgage_expense +
                lumpy_needs +
                annual_travel_desired +
                annual_gifting_desired +
                initial_estimated_tax)

            # Income (Salary, Pension, SS) and Mandatory/fixed Portfolio Draws (Trust income, 457b, RMDs)
            total_annual_cash_in = (
                salary_income +
                pension_income + 
                ss_benefit +
                def457b_income +
                trust_income +
                annual_rmd_required
            )
            mandatory_portfolio_withdrawals = def457b_income + trust_income + annual_rmd_required
            estimated_additional_withdrawal_needed = max(0.0, initial_expense_estimate - total_annual_cash_in)
            initial_portfolio_draw_estimate = mandatory_portfolio_withdrawals + estimated_additional_withdrawal_needed
            
            #print(f"STEP 3: Year {current_year}  Initial MAXIMAL Expense Estimate (without taxes)=${initial_expense_estimate:,.0f}   MAXIMAL Withdrawls=${initial_portfolio_draw_estimate:,.0f}   Income =${total_annual_cash_in:,.0f}   Est. Add. Draws=${estimated_additional_withdrawal_needed:,.0f}")

            # =========================================================================
            # --- STEP 4: Iterated Estimated TAX CALCULATION ---
            # --- Doing initial full-up estimate of taxes with ALL expenses (incl. travel and gifting)
            # =========================================================================
            # Estimate ordinary/LTCG portion for tax basis using withdrawal_engine simulation
            tax_sim_bal = copy.deepcopy(current_accounts)
            res = self.withdrawal_engine._withdraw_from_hierarchy(
                cash_needed=initial_portfolio_draw_estimate,
                accounts_bal=tax_sim_bal,
                simulate_only=True
            )
            ordinary = res.get("ordinary_inc", 0.0)
            ltcg = res.get("ltcg_inc", 0.0)
            AGI_estimate += ordinary + ltcg
            ORD_estimate += ordinary
            LTCG_estimate += ltcg
  
            state_tax, federal_tax, medicare_irmaa = calculate_taxes(
                year=current_year,
                inflation_index=current_inflation_index,
                filing_status=self.filing_status,           
                state_of_residence=self.state_of_residence, 
                age1=current_age_person1,
                age2=current_age_person2,
                magi_two_years_ago=magi_two_years_ago,
                AGI=AGI_estimate,
                taxable_ordinary=AGI_estimate,
                lt_cap_gains=LTCG_estimate,
                qualified_dividends=0.0, 
                social_security_income=ss_benefit,
            )
            iterated_estimated_tax = state_tax + federal_tax
            tax_true_up = iterated_estimated_tax - initial_estimated_tax           
            iterated_expense_estimate = initial_expense_estimate + tax_true_up

            if tax_true_up > 0: 
                # Estimate ordinary/LTCG portion of tax true up 
                tax_sim_bal = copy.deepcopy(current_accounts)
                res = self.withdrawal_engine._withdraw_from_hierarchy(
                    cash_needed=tax_true_up,
                    accounts_bal=tax_sim_bal,
                    simulate_only=True
                )
                ordinary = res.get("ordinary_inc", 0.0)
                ltcg = res.get("ltcg_inc", 0.0)
                AGI_estimate += ordinary + ltcg
                ORD_estimate += ordinary
                LTCG_estimate += ltcg
  
            #print(f"STEP 4: Year {current_year}  AGI Estimate with draw to pay taxes=${AGI_estimate:,.0f}  ITERATED MAXIMAL Expense Estimate=${iterated_expense_estimate:,.0f}  ITERATED Tax Estimates: State=${state_tax:,.0f} Federal=${federal_tax:,.0f}")

            # =========================================================================
            # --- STEP 5: Discretionary Travel
            # --- adjust travel budget if exceeding total draw limit or MAGI limit
            # =========================================================================
            # --- Define spend_plan from percentage of prior year portfolio balance ---
            prior_year_balance = self.portfolio_paths[path_index, year_index - 1]
            spend_plan = clean_percent(self.inputs.withdrawal_rate) * prior_year_balance
            self.plan_paths[path_index, year_index] = spend_plan

            # --- leftover for discretionary spending ---
            leftover_cash = max(0.0, spend_plan - iterated_expense_estimate)

            target_travel = annual_travel_desired # if current_year <= 2050 else annual_travel_desired / 2 # reduce travel late in life
            proposed_travel = min(target_travel, leftover_cash)

            # Estimate ordinary/LTCG portion for tax basis using withdrawal_engine simulation
            tax_sim_bal = copy.deepcopy(current_accounts)
            res = self.withdrawal_engine._withdraw_from_hierarchy(
                cash_needed=proposed_travel,
                accounts_bal=tax_sim_bal,
                simulate_only=True
            )
            travel_ordinary = res.get("ordinary_inc", 0.0)
            travel_ltcg = res.get("ltcg_inc", 0.0)

            AGI_proposed_travel = AGI_estimate + travel_ordinary + travel_ltcg
            
            # Use tax engine to get effective marginal rates for this income
            federal_rate, state_rate = get_effective_marginal_rates(
                year=current_year,
                income=AGI_proposed_travel,
                filing_status=self.filing_status,
                state_of_residence=self.state_of_residence,
                age1=current_age_person1,
                age2=current_age_person2,
                inflation_index=current_inflation_index
            )
            total_rate = federal_rate + state_rate

            # ----------------------------------------------------------
            # Retrieve the SAME tax/IRMAA ceilings used for ROTH Conversions
            # ----------------------------------------------------------
            tax_target_AGI, irmaa_target_AGI = get_tax_planning_targets(
                year=current_year,
                inflation_this_year=current_inflation_index,
                roth_tax_bracket=self.roth_tax_bracket,
                roth_irmaa_threshold=self.roth_irmaa_threshold,
                filing_status=self.filing_status
            )

            MAGI_limit = min(tax_target_AGI, irmaa_target_AGI)

            #print(f"STEP 5: Year {current_year} TAX Limit {tax_target_AGI:,.0f}  IRMAA Limit {irmaa_target_AGI:,.0f}  AGI with proposed Travel {AGI_proposed_travel:,.0f}")

            # Reduce proposed travel if MAGI exceeds your target threshold (if any)
            if AGI_proposed_travel > MAGI_limit:
                over = AGI_proposed_travel - MAGI_limit
                if over > 0:
                    proposed_travel -= over / total_rate
  
            # Round to nearest $1,000 and ensure non-negative
            actual_travel = max(0, math.ceil(proposed_travel / 1000) * 1000)
            self.travel_paths[path_index, year_index] = actual_travel

            if actual_travel > 0:
                # Simulate draw to update income
                res_actual = self.withdrawal_engine._withdraw_from_hierarchy(
                    cash_needed=actual_travel,
                    accounts_bal=current_accounts,
                    simulate_only=True
                )

                travel_ordinary_actual = res_actual.get("ordinary_inc", 0.0)
                travel_ltcg_actual = res_actual.get("ltcg_inc", 0.0)

                # Update AGI_estimate with income from planned travel withdrawals
                AGI_estimate += travel_ordinary_actual + travel_ltcg_actual
                ORD_estimate += travel_ordinary_actual
                LTCG_estimate += travel_ltcg_actual

            total_annual_withdrawal_needed = iterated_expense_estimate - target_travel + actual_travel - annual_gifting_desired

            #print(f"STEP 5: Year {current_year} *** NOT MAXIMAL NOW *** Total Annual Draw Needed before gifting ${total_annual_withdrawal_needed:,.0f}  actual travel ${actual_travel:,.0f}")
            
            # ----------------------------
            # STEP 6 - PLAN ROTH conversions (do NOT mutate accounts here)
            # ----------------------------
            # Determine planned conversion amounts
            total_conversion_plan = 0.0

            # Person 2 planned conversion
            p2_trad_accts = [k for k, v in current_accounts.items()
                             if v.get("owner") == "person2" and v.get("tax") == "traditional"]
            p2_trad_bal = sum(current_accounts[k]["balance"] for k in p2_trad_accts)
            conv_p2 = 0.0
            if p2_trad_bal > 0:
                conv_p2 = optimal_roth_conversion(
                    year=current_year,
                    inflation_index=current_inflation_index,
                    filing_status=self.filing_status,
                    AGI_base=AGI_estimate, # includes travel but not gifting (priorities travel -> ROTH conversions -> gifting)
                    traditional_balance=p2_trad_bal,
                    roth_tax_bracket=self.roth_tax_bracket,
                    roth_irmaa_threshold=self.roth_irmaa_threshold
                )
            conv_p2 = min(conv_p2, self.max_roth) # limit conversion to User Inoput max_roth

            # Person 1 planned conversion
            p1_trad_accts = [k for k, v in current_accounts.items()
                             if v.get("owner") == "person1" and v.get("tax") == "traditional"]
            p1_trad_bal = sum(current_accounts[k]["balance"] for k in p1_trad_accts)
            conv_p1 = 0.0
            if p1_trad_bal > 0:
                conv_p1 = optimal_roth_conversion(
                    year=current_year,
                    inflation_index=current_inflation_index,
                    filing_status=self.filing_status,
                    AGI_base=AGI_estimate + conv_p2,  # plan P1 after P2 (AGI increases if P2 converts)
                    traditional_balance=p1_trad_bal,
                    roth_tax_bracket=self.roth_tax_bracket,
                    roth_irmaa_threshold=self.roth_irmaa_threshold
                )
            conv_p1 = min(conv_p1, self.max_roth-conv_p2) # limit conversion to User Inoput max_roth         
            total_conversion_plan = conv_p1 + conv_p2

            # Keep conv_p1/conv_p2 locally so we can execute them in Q4 later.
            planned_conv = {"person1": conv_p1, "person2": conv_p2}
            
            AGI_estimate += total_conversion_plan
            
            #print(f"STEP 6: Year {current_year} AGI with planned ROTH ${AGI_estimate:,.0f}  Planned ROTH Conversion ${total_conversion_plan:,.0f}")
            
            # =========================================================================
            # --- STEP 7: Final Estimated TAX CALCULATION ---
            # --- Doing iteration factoring in planned ROTH conversions now
            # =========================================================================
            state_tax, federal_tax, medicare_irmaa = calculate_taxes(
                year=current_year,
                inflation_index=current_inflation_index,
                filing_status=self.filing_status,           
                state_of_residence=self.state_of_residence, 
                age1=current_age_person1,
                age2=current_age_person2,
                magi_two_years_ago=magi_two_years_ago,
                AGI=AGI_estimate,
                taxable_ordinary=ORD_estimate,
                lt_cap_gains=LTCG_estimate,
                qualified_dividends=0.0, 
                social_security_income=ss_benefit,
            )
            final_estimated_tax = state_tax + federal_tax

            #print(f"STEP 7: Year {current_year}  AGI Estimate with planned ROTH conversions=${AGI_estimate:,.0f}  FINAL Tax Estimates: State=${state_tax:,.0f} Federal=${federal_tax:,.0f}")

            # =========================================================================
            # --- STEP 8: Final Portfolio Draw CALCULATION ---
            # --- gifting is excluded and handled seperately in Q4
            # =========================================================================
            final_expense_estimate = (
                annual_base_spending +
                mortgage_expense +
                lumpy_needs +
                actual_travel +
                final_estimated_tax)

            final_additional_withdrawal_needed = max(0.0, final_expense_estimate - total_annual_cash_in)

            # =========================================================================
            # --- STEP 9: QUARTERLY WITHDRAWAL AND INVESTMENT RETURNS ---
            # =========================================================================

            # Slice the 4 quarterly returns
            eq_q_returns   = equity_q_path[i * 4 : (i + 1) * 4]
            bond_q_returns = bond_q_path[i * 4 : (i + 1) * 4]
            actual_withdrawals = 0.0

            # Track realized tax character for this year's income accounting

            for q in range(4):
                #  Deepcopy for tax simulation (already in your code)
                tax_sim_bal = copy.deepcopy(current_accounts)

                # ---------------------------------------------------------------------
                # 1) MANDATORY QUARTERLY DRAWS (RMDs & Trust Income) and 457b Income
                # ---------------------------------------------------------------------
                
                # We no longer rely on a total RMD target, but withdraw the 1/4 RMD
                # from each specific account, as required by law.
                
                # 1a) RMD Draw: Withdraw the required amount from each specific account
                quarterly_rmd_total = 0.0
                for acct_name, annual_rmd_amount in rmds_by_account_annual.items():
                    # Look up the account state
                    acct = current_accounts.get(acct_name)
                    
                    if acct and acct.get("balance", 0.0) > 0:
                        
                        quarterly_rmd = annual_rmd_amount / 4.0
                        
                        # Draw amount is limited by the current balance
                        draw_amount = min(quarterly_rmd, acct["balance"])
                        
                        acct["balance"] -= draw_amount
                        quarterly_rmd_total += draw_amount
                        final_ordinary_income_actual += draw_amount # RMD is ordinary income

                # 1b) Trust Income Draw: Withdraw the quarterly portion from Trust accounts
                quarterly_trust_income_draw = trust_income / 4.0
                trust_draw_remaining_q = quarterly_trust_income_draw

                # Iterate over accounts, prioritizing drawing the required trust income
                # from any available Trust account until the quarterly requirement is met.
                for acct_name, acct in current_accounts.items():
                    if acct.get("tax") == "trust" and acct.get("balance", 0.0) > 0 and trust_draw_remaining_q > 0:
                        
                        # Draw up to the remaining required trust income, capped by balance
                        draw_amount = min(trust_draw_remaining_q, acct["balance"])
                        
                        acct["balance"] -= draw_amount
                        trust_draw_remaining_q -= draw_amount
                        final_ordinary_income_actual += draw_amount # Trust Income is ordinary income

                        if trust_draw_remaining_q <= 0:
                            break # Trust income draw satisfied for the quarter
                        
                 # 1c) Deferred 457b Income Draw: Withdraw the quarterly portion from 457b accounts
                quarterly_def457b_income_draw = def457b_income / 4.0
                def457b_draw_remaining_q = quarterly_def457b_income_draw

                for acct_name, acct in current_accounts.items():
                    if acct.get("tax") == "def457b" and acct.get("balance", 0.0) > 0 and def457b_draw_remaining_q > 0:
                        
                        draw_amount = min(def457b_draw_remaining_q, acct["balance"])
                        
                        acct["balance"] -= draw_amount
                        def457b_draw_remaining_q -= draw_amount
                        final_ordinary_income_actual += draw_amount # Trust Income is ordinary income
                         
                        if def457b_draw_remaining_q <= 0:
                            break # def457b income draw satisfied for the quarter

                # ---------------------------------------------------------------------
                # 2) RESIDUAL PORTFOLIO WITHDRAWAL (For remaining spending needs)
                # --- amouint needed calculatyed in Step 8 above
                # ---------------------------------------------------------------------
                quarterly_portfolio_draw_needed = (final_additional_withdrawal_needed) / 4.0 

                quarterly_portfolio_draw_remaining = quarterly_portfolio_draw_needed
                if quarterly_portfolio_draw_remaining > 0:
                    withdrawal_result = self.withdrawal_engine._withdraw_from_hierarchy(
                        cash_needed=quarterly_portfolio_draw_remaining,
                        accounts_bal=current_accounts,
                        simulate_only=False,
                    )
                    actual_withdrawals += quarterly_portfolio_draw_remaining
                    final_ordinary_income_actual += withdrawal_result.get("ordinary_inc", 0.0)
                    final_ltcg_income_actual += withdrawal_result.get("ltcg_inc", 0.0)

                # ---------------------------------------------------------------------
                # 3) In Q4 do ROTH conversions and Gifting
                # ---------------------------------------------------------------------
                if q == 3:
                    tax_sim_bal = copy.deepcopy(current_accounts)
                    # Execute conversions in Q4 only
                    running_total = 0.0
                    # Person 2
                    conv = planned_conv.get("person2", 0.0)
                    if conv > 0:
                        # pro-rata across person's trad accounts
                        p2_trads = [n for n, a in current_accounts.items() if a.get("owner") == "person2" and a.get("tax") == "traditional" and a.get("balance",0) > 0]
                        p2_total = sum(current_accounts[n]["balance"] for n in p2_trads)
                        # choose a Roth target (existing roth)
                        p2_roths = [n for n,a in current_accounts.items() if a.get("owner") == "person2" and a.get("tax") == "roth"]
                        if p2_roths and p2_total > 0:
                            roth_target = p2_roths[0]
                            remaining = conv
                            for n in p2_trads:
                                if remaining <= 0:
                                    break
                                available = current_accounts[n]["balance"]
                                take = min(available, conv * (available / p2_total))
                                # safety clamp
                                take = min(take, remaining, current_accounts[n]["balance"])

                                available = current_accounts[n]["balance"]
                                # 1. Calculate the pro-rata share of the total conversion target (conv)
                                pro_rata_share = conv * (available / p2_total)
                                # 2. The amount to take is the minimum of the calculated share,
                                # the remaining need, and the available balance.
                                take = min(pro_rata_share, remaining, available)

                                current_accounts[n]["balance"] -= take
                                current_accounts[roth_target]["balance"] += take
                                final_ordinary_income_actual += take   # conversions = ordinary income realized in Q4
                                remaining -= take
                                running_total += take

                    # Person 1 (same pattern)
                    conv = planned_conv.get("person1", 0.0)
                    if conv > 0:
                        p1_trads = [n for n, a in current_accounts.items() if a.get("owner") == "person1" and a.get("tax") == "traditional" and a.get("balance",0) > 0]
                        p1_total = sum(current_accounts[n]["balance"] for n in p1_trads)
                        p1_roths = [n for n,a in current_accounts.items() if a.get("owner") == "person1" and a.get("tax") == "roth"]
                        if p1_roths and p1_total > 0:
                            roth_target = p1_roths[0]
                            remaining = conv
                            for n in p1_trads:
                                if remaining <= 0:
                                    break
                                available = current_accounts[n]["balance"]
                                # 1. Calculate the pro-rata share of the total conversion target (conv)
                                pro_rata_share = conv * (available / p1_total)
                                # 2. The amount to take is the minimum of the calculated share,
                                # the remaining need, and the available balance.
                                take = min(pro_rata_share, remaining, available)

                                current_accounts[n]["balance"] -= take
                                current_accounts[roth_target]["balance"] += take
                                final_ordinary_income_actual += take
                                remaining -= take
                                running_total += take
                                
                    # after both person conv executions
                    self.conversion_paths[path_index, year_index] = running_total

                    # -------------------------------------
                    # 2) Gifting — WITH TAX + IRMAA LIMITS
                    # -------------------------------------

                    # Dollar target (already inflation adjusted)
                    target_gifting = annual_gifting_desired

                    # ----------------------------------------------------------
                    # Step B: Compute tax impact of gifting via SIMULATED draw
                    # ----------------------------------------------------------
                    #tax_sim_bal = copy.deepcopy(current_accounts)

                    sim_result = self.withdrawal_engine._withdraw_from_hierarchy(
                        cash_needed=target_gifting,
                        accounts_bal=tax_sim_bal,
                        simulate_only=True
                    )

                    ordinary_income = sim_result.get("ordinary_inc", 0.0)
                    ltcg_income = sim_result.get("ltcg_inc", 0.0)

                    # This is the AGI/MAGI AFTER gifting
                    taxable_after_gifting = AGI_estimate + ordinary_income + ltcg_income

                    max_room = max(0.0, MAGI_limit -AGI_estimate) 

                    # ----------------------------------------------------------
                    # Step D: Adjust gifting so it does NOT exceed tax/IRMAA room
                    # ----------------------------------------------------------
                    income_added = ordinary_income + ltcg_income

                    if income_added > max_room:
                        excess_income = income_added - max_room

                        # Calculate the average percentage of the withdrawal that is taxable income.
                        # We must be defensive against gifting being zero.
                        if target_gifting > 0:
                            avg_taxable_pct = income_added / target_gifting

                            # Calculate the required reduction in the *withdrawal* amount to cut the 
                            # *income* by the excess_income amount.
                            if avg_taxable_pct > 0:
                                reduction_in_gifting = excess_income / avg_taxable_pct
                                adjusted_gifting = target_gifting - reduction_in_gifting
                            else:
                                # If no income was added (e.g., all from Roth/Basis), no adjustment needed
                                adjusted_gifting = target_gifting
                        else:
                            # If gifting was 0, it stays 0
                            adjusted_gifting = 0.0
                    else:
                        adjusted_gifting = target_gifting
                        
                    # ----------------------------------------------------------
                    # Step E: Final rounding logic (consistent with Roth)
                    # ----------------------------------------------------------
                    if adjusted_gifting < 1000:
                        actual_gifting = max(0.0, adjusted_gifting)
                    else:
                        actual_gifting = max(0.0, round(adjusted_gifting, -3))

                    self.gifting_paths[path_index, year_index] = actual_gifting

                    # ----------------------------------------------------------
                    # Step F: ACTUAL GIFTING WITHDRAWAL EXECUTION
                    # true up taxes before we take final portfolio draw
                    # ----------------------------------------------------------
                    final_AGI_actual = final_ordinary_income_actual + final_ltcg_income_actual
                    if actual_gifting > 0:
                        withdrawal_result = self.withdrawal_engine._withdraw_from_hierarchy(
                            cash_needed=actual_gifting,
                            accounts_bal=current_accounts,
                            simulate_only=True 
                        )
                        final_ordinary_income_actual += withdrawal_result.get("ordinary_inc", 0.0)
                        final_ltcg_income_actual += withdrawal_result.get("ltcg_inc", 0.0)
                        final_AGI_actual = final_ordinary_income_actual + final_ltcg_income_actual

                    state_tax, federal_tax, medicare_irmaa = calculate_taxes(
                        year=current_year,
                        inflation_index=current_inflation_index,
                        filing_status=self.filing_status,           
                        state_of_residence=self.state_of_residence, 
                        age1=current_age_person1,
                        age2=current_age_person2,
                        magi_two_years_ago=magi_two_years_ago,
                        AGI=final_AGI_actual,
                        taxable_ordinary=final_ordinary_income_actual,
                        lt_cap_gains=final_ltcg_income_actual,
                        qualified_dividends=0.0,
                        social_security_income=ss_benefit,
                    )
                    post_gifting_estimated_tax = state_tax + federal_tax
                    tax_true_up = post_gifting_estimated_tax - final_estimated_tax

                    final_portfolio_draw = actual_gifting + tax_true_up
                    withdrawal_result = self.withdrawal_engine._withdraw_from_hierarchy(
                        cash_needed=final_portfolio_draw,
                        accounts_bal=current_accounts,
                        simulate_only=False
                    )
 
                    actual_withdrawals += final_portfolio_draw
                    
                # for plotting RMDs, Trust income and 457b income are displayed sperately so not included
                # Needs to be outside quarterly loop
                self.portfolio_withdrawal_paths[path_index, year_index] = actual_withdrawals

                # ---------------------------------------------------------------------
                # 3) APPLY QUARTERLY RETURNS
                # ---------------------------------------------------------------------
                eq_r = eq_q_returns[q]
                bond_r = bond_q_returns[q]

                for acct_name, acct in current_accounts.items():

                    bal = acct.get("balance", 0.0)
                    if bal <= 0:
                        # Skip negative/empty accounts — prevents pathological return explosions
                        acct["balance"] = 0.0
                        continue

                    # Guaranteed normalized in Stage 1
                    eq_pct = acct["equity_pct"]
                    bond_pct = acct["bond_pct"]

                    blended_q_ret = eq_pct * eq_r + bond_pct * bond_r

                    # Apply return
                    new_bal = bal * (1 + blended_q_ret)
                    acct["balance"] = new_bal

                    # Basis tracking for taxable accounts only
                    if acct.get("tax") == "taxable":
                        basis = acct.get("basis", 0.0)
                        acct["basis"] = basis + (basis * blended_q_ret)

            # =========================================================================
            # --- STEP 10: POST-QUARTERLY CLEANUP AND SAVE DATA ---
            # =========================================================================
            # final_ordinary_income_actual and final_ltcg_income_actual include:
            #   - withdrawals that were ordinary income
            #   - realized LTCG
            #   - ROTH conversions we added in Q4 above (as ordinary income)

            AGI = final_ordinary_income_actual + final_ltcg_income_actual
            state_tax_final, federal_tax_final, medicare_irmaa_final = calculate_taxes(
                year=current_year,
                inflation_index=current_inflation_index,
                filing_status=self.filing_status,
                state_of_residence=self.state_of_residence,
                age1=current_age_person1,
                age2=current_age_person2,
                magi_two_years_ago=self.magi_paths[path_index, year_index - 2],
                AGI=AGI,
                taxable_ordinary=final_ordinary_income_actual,
                lt_cap_gains=final_ltcg_income_actual,
                qualified_dividends=0.0,
                social_security_income=ss_benefit,
            )
            total_taxes_final = state_tax_final + federal_tax_final
            self.taxes_paths[path_index, year_index] = total_taxes_final
            self.medicare_paths[path_index, year_index] = medicare_irmaa_final

            #print(f"STEP 9b: Year {current_year} FED {federal_tax_final:,.0f} VA {state_tax_final:,.0f}  AGI {AGI:,.0f} ORDINARY {final_ordinary_income_actual:,.0f} LTCG {final_ltcg_income_actual:,.0f}  SS {ss_benefit:,.0f}")

            ss_taxable = self.accounts_income.compute_taxable_ss(ss_benefit, AGI, self.inputs.filing_status)
            MAGI = AGI + ss_taxable
            self.magi_paths[path_index, year_index] = MAGI

            # Final Portfolio Balance
            self.portfolio_paths[path_index, year_index] = self._get_total_portfolio(current_accounts)
            
            # Save the final account state for the year
            self.account_paths[path_index][year_index] = copy.deepcopy(current_accounts)

            # CHECKSUM - total expense - total income
            checksum = 0.0
            total_expense = 0.0
            final_cash_in = 0.0
            total_expenses = annual_base_spending + mortgage_expense + lumpy_needs +  total_taxes_final + actual_travel + actual_gifting
            final_cash_in = salary_income + pension_income + def457b_income + trust_income + ss_benefit + annual_rmd_required + actual_withdrawals
            checksum = total_expenses - final_cash_in

            #print(f"STEP 10: Year {current_year} checksum {checksum:,.0f} taxable_SS {ss_taxable:,.0F}  FINAL expenses {total_expenses:,.0f}  FINAL cash in {final_cash_in:,.0f}   actual gifting ${actual_gifting:,.0f} annual withdrawal needed ${final_additional_withdrawal_needed:,.0f}   actual withdrawls ${actual_withdrawals:,.0f}\n")
            

    # =========================================================================
    # 3. UTILITY FUNCTIONS — Modular Wiring Only
    # =========================================================================

    def _normalize_accounts(self):
        """Ensure account dicts have required fields and canonical numeric types.

        - Accept either 'equity'/'bond' or 'equity_pct'/'bond_pct' from XML.
        - Convert numeric strings to floats.
        - Ensure balances/basis are floats (default 0.0).
        - Compute equity_pct as equity / (equity + bond) if both present.
        """
        import numbers

        # Make a defensive deep copy of the input accounts so we can mutate safely
        self.initial_accounts = copy.deepcopy(self.inputs.accounts)

        for acct_name, acct in self.initial_accounts.items():
            # --- normalize numeric fields ---
            # Accept strings coming from XML and coerce to float safely.
            def _to_float(val, default=0.0):
                if val is None:
                    return default
                if isinstance(val, numbers.Number):
                    return float(val)
                try:
                    return float(str(val))
                except Exception:
                    return default

            acct['balance'] = _to_float(acct.get('balance', 0.0), 0.0)
            acct['basis']   = _to_float(acct.get('basis', 0.0), 0.0)

            # Tax/owner defaults
            acct['tax'] = acct.get('tax') or 'traditional'
            acct['owner'] = acct.get('owner') or 'person1'
            acct.setdefault('income', 0.0)
            acct.setdefault('ordinary_pct', 0.1)

            # --- normalize asset mix ---
            # Accept either 'equity'/'bond' or 'equity_pct'/'bond_pct'
            raw_equity = acct.get('equity', acct.get('equity_pct', None))
            raw_bond   = acct.get('bond', acct.get('bond_pct', None))

            # Coerce to floats where present
            equity = _to_float(raw_equity, None)
            bond   = _to_float(raw_bond, None)

            # If both are None => fallback default mix (70/30)
            if equity is None and bond is None:
                equity_pct = 0.70
            else:
                # If only one side present, assume the other is the complement (if sensible)
                if equity is None:
                    # bond given -> equity = 1 - bond
                    bond = min(max(bond, 0.0), 1.0)
                    equity = 1.0 - bond
                elif bond is None:
                    equity = min(max(equity, 0.0), 1.0)
                    bond = 1.0 - equity

                s = equity + bond
                if s <= 0:
                    equity_pct = 0.70
                else:
                    equity_pct = equity / s

            acct['equity_pct'] = float(equity_pct)
            acct['bond_pct'] = float(1.0 - equity_pct)
            
         # End _normalize_accounts

    def _get_total_portfolio(self, accounts: Dict) -> float:
        """Return total portfolio balance as sum of numeric balances (no magic pennies)."""
        total = 0.0
        for acct in accounts.values():
            bal = acct.get('balance', 0.0)
            try:
                total += float(bal) if bal is not None else 0.0
            except Exception:
                # defensive fallback if someone passed a weird type
                total += 0.0
        return total


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
            n_full=self.n_full,
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
        cumulative = [1.0] * self.n_full
        for y in range(self.n_full):
            q_rates = infl_q[0, y*4:(y+1)*4]
            ann_rate, _ = calculate_annual_inflation(q_rates, cumulative[-1])
            if y >= 2:
                cumulative[y] = cumulative[y-1] * (1 + ann_rate)
            else:
                cumulative[y] = cumulative[y-1]
        return cumulative

    def _calculate_annual_spending_needs(self, 
        current_year: int, 
        inflation_index: float) -> Tuple[float, float, float, float, float, float]:
        """
        Calculates total annual expenses (fixed and adjustable).
        
        Returns: 
            (annual_base_spending, mortgage_expense, lumpy_needs,  
             annual_travel_desired, annual_gifting_desired)
        """
        
        from config.expense_assumptions import (
            mortgage_payoff_year, mortgage_monthly_until_payoff,
            property_tax_and_insurance, car_replacement_cycle,
            car_cost_today, lumpy_expenses,
            home_repair_prob, home_repair_mean, home_repair_shape,
        )

        #print(f"Morgage year {mortgage_payoff_year}  Mortgage {mortgage_monthly_until_payoff:,.2f}  T&I {property_tax_and_insurance:,.2f}")
        
        # 1. Base Spending (Inflation-Adjusted)
        annual_base_spending = self.base_annual_spending * inflation_index
        
        monthly_mortgage = 0.0
        if current_year <= mortgage_payoff_year:
            monthly_mortgage = mortgage_monthly_until_payoff # This is not subject to inflation          
        taxes_and_insurance = property_tax_and_insurance * inflation_index
        mortgage_expense = 12 * monthly_mortgage + taxes_and_insurance
        

        # 2. Lumpy Expenses (Check against configuration list)
        lumpy_needs = 0.0
        for item in lumpy_expenses:
            if item.get("year") == current_year:
                lumpy_needs += item.get("amount", 0.0) * inflation_index 
                
        # Car Replacement (Every 'car_replacement_cycle' years, inflated)
        car_expense = 0.0
        years_since_start = current_year - self.current_year - 2 #offset so first car expense out 2 years
        if years_since_start % car_replacement_cycle == 0 and years_since_start >= 0:
            car_expense = car_cost_today * inflation_index
        lumpy_needs += car_expense
            
        # Stochastic Home Repair (Restored Logic)
        home_repair = 0.0
        if self.rng.random() < home_repair_prob:
            # Draw from log-normal distribution (requires self.rng to be a seeded generator)
            mu_log = np.log(home_repair_mean) - (home_repair_shape ** 2) / 2
            home_repair = self.rng.lognormal(mu_log, home_repair_shape) * inflation_index
        lumpy_needs += home_repair

        #print(f"Home Repair {home_repair:,.2f}  Mortgage {mortgage_expense:,.2f}")


        # 6. Discretionary Spending (Dynamically Adjustable to manage tax/IRMAA)
        # These are the DESIRED amounts, which may be cut by the planner (Step 2/3)
        annual_travel_desired = self.travel * inflation_index
        annual_gifting_desired = self.gifting * inflation_index
        
        return (
            annual_base_spending,
            mortgage_expense,
            lumpy_needs,
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
                "salary_paths": np.array([]),
                "def457b_income_paths": np.array([]),
                "pension_paths": np.array([]),
                "medicare_paths": np.array([]),
                "travel_paths": np.array([]),
                "gifting_paths": np.array([]),
                "base_spending_paths": np.array([]),
                "mortgage_expense_paths": np.array([]),
                "lumpy_spending_paths": np.array([]),
                "plan_paths": np.array([]),
             }

        success = self.inputs.success_threshold
        avoid_ruin = self.inputs.avoid_ruin_threshold
        portfolio_end = self.portfolio_paths[:, -1]
        
        success_rate = np.mean(portfolio_end > success) * 100
        minimum_annual_balance = np.min(self.portfolio_paths, axis=1)
        avoid_ruin_rate = np.mean(minimum_annual_balance > avoid_ruin) * 100

        # Transpose account_paths and filter to XML-defined names ---
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
            "salary_paths": getattr(self, 'salary_paths', np.array([])),
            "def457b_income_paths": getattr(self, 'def457b_income_paths', np.array([])),
            "pension_paths": getattr(self, 'pension_paths', np.array([])),
            "medicare_paths": getattr(self, 'medicare_paths', np.array([])),
            "travel_paths": getattr(self, 'travel_paths', np.array([])),
            "gifting_paths": getattr(self, 'gifting_paths', np.array([])),
            "base_spending_paths": getattr(self, 'base_spending_paths', np.array([])),
            "mortgage_expense_paths": getattr(self, 'mortgage_expense_paths', np.array([])),
            "lumpy_spending_paths": getattr(self, 'lumpy_spending_paths', np.array([])),
            "plan_paths": getattr(self, 'plan_paths', np.array([])),
        }
        return result
