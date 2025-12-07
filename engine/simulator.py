import numpy as np
import pandas as pd
import copy
import math
from typing import List, Dict, Tuple, Any

from utils.xml_loader import DEFAULT_SETUP, DEFAULT_ACCOUNTS

from models import PlannerInputs

from config.expense_assumptions import *
from config.market_assumptions import *

from engine.rmd_tables import get_rmd_factor
from engine.def457b_tables import get_def457b_factor
from engine.roth_optimizer import optimal_roth_conversion
from engine.tax_planning import *
from engine.tax_engine import *
from engine.market_generator import *
from engine.income_calculator import *

class RetirementSimulator:
    """
    Runs Monte Carlo simulations for retirement planning, calculating taxes 
    and optimizing Roth conversions based on user inputs.
    """
    def __init__(self, inputs: PlannerInputs):
        self.inputs = inputs
        
        # Set all input fields as class attributes
        for field, value in inputs.__dict__.items():
            setattr(self, field, value)
             
        # Use the inputs.accounts dictionary which plotting.py references for metadata
        accounts_dict = self.inputs.accounts 

        self._normalize_accounts() 

        # Initialize ages based on birth year
        self.current_age_p1 = self.current_year - self.person1_birth_year
        self.current_age_p2 = self.current_year - (self.person2_birth_year or self.person1_birth_year) # don't allow None
        self.num_years = self.end_age - min(self.current_age_p1, self.current_age_p2)
        
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
        self.travel_paths = None # Assuming this was part of your original paths

        self.debug_log = []


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
        self.medicare_paths = np.zeros((num_paths, num_years))
        self.rmd_paths = np.zeros((num_paths, num_years))
        self.def457b_income_paths = np.zeros((num_paths, num_years))
        self.pension_paths = np.zeros((num_paths, num_years))
        self.ssbenefit_paths = np.zeros((num_paths, num_years))
        self.portfolio_withdrawal_paths = np.zeros((num_paths, num_years))
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
        
        # Initialize path-specific history arrays
        # MAGI history needs to be long enough to track 2 years prior for IRMAA (index offset)
        magi_path = [0.0] * 2 + [0.0] * self.num_years
        
        market_path = self._generate_market_path()
        inflation_path = self._generate_inflation_path()
        
        for i in range(self.num_years):
            year_index = i
            current_year = self.current_year + i
            current_inflation_index = inflation_path[i]
            
            # Update ages
            current_age_person1 = self.current_age_p1 + i
            current_age_person2 = self.current_age_p2 + i if self.current_age_p2 is not None else None
            
            # --- STEP 1: CALCULATE ANNUAL INCOME AND RMDs ---
            
            # AGI is calculated progressively
            AGI = 0.0
            
            # RMDs (Required Minimum Distributions)
            rmd_income = self._calculate_rmds(current_accounts, current_age_person1, current_age_person2, current_year)
            AGI += rmd_income
            self.rmd_paths[path_index, i] = rmd_income
            
            # Pension/Def457b Income
            pension_income = self._get_pension_benefit(current_year, current_inflation_index)
            def457b_income = self._get_def457b_income(current_year, current_inflation_index)
            AGI += pension_income + def457b_income
            self.pension_paths[path_index, i] = pension_income
            self.def457b_income_paths[path_index, i] = def457b_income
            
            # Social Security Income
            ss_benefit = self._get_social_security_benefit(current_year, current_inflation_index)
            ss_taxable, ss_non_taxable = self._calculate_taxable_ss(ss_benefit, AGI) 
            AGI += ss_taxable # Taxable portion of SS contributes to AGI
            self.ssbenefit_paths[path_index, i] = ss_benefit # Save full benefit amount
            
            # --- STEP 2: ROTH CONVERSION OPTIMIZATION ---
            
            conversion_amount = optimal_roth_conversion(
                year=current_year,
                inflation_index=current_inflation_index,
                AGI_base=AGI, # AGI before conversion
                accounts=current_accounts,
                tax_strategy=self.tax_strategy,
                irmaa_strategy=self.irmaa_strategy,
                filing_status=self.filing_status # FIXED: Passed filing status
            )
            
            # Execute conversion (removes from Traditional, adds to Roth, increases AGI)
            current_accounts.setdefault("Traditional", {"balance": 0.0})["balance"] = max(0, current_accounts.get("Traditional", {}).get("balance", 0.0) - conversion_amount)
            current_accounts.setdefault("Roth", {"balance": 0.0})["balance"] += conversion_amount
            AGI += conversion_amount
            self.conversion_paths[path_index, i] = conversion_amount
            
            # --- STEP 3: TAX CALCULATION ---
            
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
            
            # --- STEP 4: WITHDRAWALS AND EXPENSES ---
            
            cash_needed = self._get_cash_needed(current_year, current_inflation_index)
            cash_needed += total_taxes 
            
            withdrawal_order = self._get_withdrawal_order(current_year)
            # ordinary_w and ltcg_w are not used for taxes in this loop (only in dry-run)
            ordinary_w, ltcg_w = self._execute_withdrawals(current_accounts, cash_needed, withdrawal_order)
            self.portfolio_withdrawal_paths[path_index, i] = cash_needed

            # --- STEP 5: END-OF-YEAR PORTFOLIO GROWTH ---
            self._apply_market_returns(current_accounts, market_path[i])
            
            # --- STEP 6: SAVE PATH DATA ---
            self.portfolio_paths[path_index, i] = self._get_total_portfolio(current_accounts)
            self.taxes_paths[path_index, i] = total_taxes
            self.magi_paths[path_index, i] = MAGI
            self.medicare_paths[path_index, i] = medicare_irmaa
            
            # Save the final account state for the year
            self.account_paths[path_index][i] = copy.deepcopy(current_accounts)

            # Update MAGI history for the next iteration's IRMAA calculation
            magi_path[year_index + 2] = MAGI


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
            acct.setdefault("start_age", 60)
            acct.setdefault("drawdown_years", 5)
            acct.setdefault("mandatory_yield", 0.0)
            acct.setdefault("ordinary_pct", 0.1)

    def _get_total_portfolio(self, accounts: Dict) -> float:
        return sum(acct.get("balance", 0.0) for acct in accounts.values())

    def _generate_market_path(self) -> List[float]:
        """One path of annual portfolio returns using market_generator."""
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
        # 70/30 blend → annual returns
        annual = 0.7 * eq_q[0, ::4] + 0.3 * bond_q[0, ::4]
        return annual.tolist()

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

        threshold = self.target_portfolio_value 
        portfolio_end = self.portfolio_paths[:, -1]
        
        success_rate = np.mean(portfolio_end > threshold) * 100
        minimum_annual_balance = np.min(self.portfolio_paths, axis=1)
        avoid_ruin_rate = np.mean(minimum_annual_balance > 0.0) * 100
        
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
