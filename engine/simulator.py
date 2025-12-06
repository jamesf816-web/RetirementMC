import numpy as np
import pandas as pd
import copy
import math
from typing import List, Dict, Tuple, Any

from utils.xml_loader import DEFAULT_SETUP, DEFAULT_ACCOUNTS

# FIXED: Explicitly importing constants from the assumption files
from config.expense_assumptions import (
    medicare_start_age,
    medicare_part_b_base_2026,
    medicare_supplement_annual,
    mortgage_payoff_year,
    mortgage_monthly_until_payoff,
    property_tax_and_insurance,
    car_replacement_cycle,
    car_cost_today,
    car_inflation,
    lumpy_expenses,
    irmaa_brackets_start_year,
    home_repair_prob, 
    home_repair_mean, 
    home_repair_shape,
)
from config.market_assumptions import (
    initial_inflation_mu,
    initial_inflation_sigma,
    long_term_inflation_mu,
    long_term_inflation_sigma,
    years_to_revert,
    initial_equity_mu,
    initial_equity_sigma,
    initial_bond_mu,
    initial_bond_sigma,
    corr_matrix,
)

from models import PlannerInputs

from engine.rmd_tables import get_rmd_factor
from engine.def457b_tables import get_def457b_factor
from engine.roth_optimizer import optimal_roth_conversion
from engine.tax_engine import calculate_taxes


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

        # --- ROBUSTNESS CHECK FOR CORE SIMULATION PARAMETERS ---
        # Ensure critical attributes are set, providing defaults if missing from PlannerInputs
        if not hasattr(self, 'current_year') or self.current_year is None:
            self.current_year = 2025 
        if not hasattr(self, 'end_year') or self.end_year is None:
            # Calculate end year based on end_age (assuming end_age is in PlannerInputs)
            end_age = getattr(self, 'end_age', 95)
            self.end_year = self.current_year + (end_age - (self.current_year - self.person1_birth_year))
            
        # FIX: Ensure nsims is set (it's the correct parameter name from models.py)
        if not hasattr(self, 'nsims') or self.nsims is None:
            self.nsims = 100 # Default to 100 paths
            
        if not hasattr(self, 'filing_status') or self.filing_status is None:
            self.filing_status = "married_joint"
        
        if not hasattr(self, 'state_of_residence') or self.state_of_residence is None:
            self.state_of_residence = "VA"

        if not hasattr(self, 'target_portfolio_value') or self.target_portfolio_value is None:
             self.target_portfolio_value = 1000000.0
             
        COMMON_ACCOUNT_DEFAULTS = {
            "Roth": {"tax": "exempt", "owner": "person1"},
            "Traditional": {"tax": "deferred", "owner": "person1"},
            "Taxable": {"tax": "taxable", "owner": "joint"},
        }
        
        # Use the inputs.accounts dictionary which plotting.py references for metadata
        accounts_dict = self.inputs.accounts 

        for name, defaults in COMMON_ACCOUNT_DEFAULTS.items():
            if name not in accounts_dict:
                accounts_dict[name] = {}
            
            # Ensure the critical 'tax' and 'owner' keys are present
            accounts_dict[name].setdefault("tax", defaults["tax"])
            accounts_dict[name].setdefault("owner", defaults["owner"])
            
        self._normalize_accounts() 

        # Initialize ages based on birth year
        self.person1_age = self.current_year - self.person1_birth_year
        self.person2_age = self.current_year - self.person2_birth_year if self.person2_birth_year is not None else None
        
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
        num_years = self.end_year - self.current_year + 1
        num_paths = self.nsims 
        
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
        
        # Initialize state for this path
        current_accounts = copy.deepcopy(self.initial_accounts)
        
        # Initialize path-specific history arrays
        # MAGI history needs to be long enough to track 2 years prior for IRMAA (index offset)
        num_years = self.end_year - self.current_year + 1
        magi_path = [0.0] * 2 + [0.0] * num_years
        
        market_path = self._generate_market_path()
        inflation_path = self._generate_inflation_path()
        
        for i in range(num_years):
            year_index = i
            current_year = self.current_year + i
            current_inflation_index = inflation_path[i]
            
            # Update ages
            current_age_person1 = self.person1_age + i
            current_age_person2 = self.person2_age + i if self.person2_age is not None else None
            
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
    # 3. UTILITY FUNCTIONS (Restored to their original/placeholder form)
    # =========================================================================
    
    def _normalize_accounts(self):
        """Initializes account structure if missing fields are found."""
        if not hasattr(self, 'accounts') or not self.accounts:
            self.initial_accounts = DEFAULT_ACCOUNTS
        else:
            self.initial_accounts = self.accounts
        
        # Ensure all accounts have a 'balance' key
        for name, acct in self.initial_accounts.items():
            acct.setdefault('balance', 0.0)
            acct.setdefault('basis', 0.0) # Used for taxable gains calculation

    def _get_total_portfolio(self, accounts: Dict) -> float:
        """Calculates total portfolio value."""
        return sum(acct.get('balance', 0.0) for acct in accounts.values())

    def _generate_market_path(self) -> List[float]:
        """Generates a sequence of annualized portfolio returns for one path."""
        num_years = self.end_year - self.current_year + 1
        # Complex logic using corr_matrix, initial_equity_mu, etc. would go here.
        # Returning a simplified random walk for compilation/execution
        return np.random.normal(0.06, 0.12, num_years)

    def _generate_inflation_path(self) -> List[float]:
        """Generates the cumulative inflation index for one path."""
        num_years = self.end_year - self.current_year + 1
        # Complex logic using long_term_inflation_mu, years_to_revert, etc. would go here.
        # Returning a simplified cumulative index
        inflation_rates = np.random.normal(0.03, 0.01, num_years)
        return np.cumprod(1 + inflation_rates)

    def _get_social_security_benefit(self, year: int, inflation_index: float) -> float:
        """Calculates the total SS benefit for the year."""
        # This needs to incorporate self.person1_ss_fra, self.person2_ss_fra, etc.
        return 0.0 

    def _calculate_taxable_ss(self, ss_benefit: float, AGI: float) -> Tuple[float, float]:
        """Calculates taxable and non-taxable portions of SS benefit."""
        # Requires complex Provisional Income calculation logic
        return ss_benefit * 0.85, ss_benefit * 0.15 

    def _calculate_rmds(self, accounts: Dict, age1: int, age2: int, year: int) -> float:
        """Calculates and returns total RMD income."""
        # Requires RMD table lookup using get_rmd_factor
        return 0.0

    def _get_pension_benefit(self, year: int, inflation_index: float) -> float:
        """Calculates pension income."""
        return 0.0

    def _get_def457b_income(self, year: int, inflation_index: float) -> float:
        """Calculates 457b income."""
        # Requires def457b_tables.get_def457b_factor
        return 0.0

    def _get_cash_needed(self, year: int, inflation_index: float) -> float:
        """Calculates annual spending including base, mortgage, and lumpy expenses."""
        # This needs to incorporate the imported expense assumptions
        base_spending = getattr(self, 'base_annual_spending', 50000.0)
        return base_spending * inflation_index

    def _get_withdrawal_order(self, year: int) -> List[str]:
        """Returns the strategic withdrawal order (e.g., Taxable, Traditional, Roth)."""
        return ["Taxable", "Traditional", "Roth"]

    def _execute_withdrawals(self, accounts: Dict, amount: float, order: List[str]) -> Tuple[float, float]:
        """Executes withdrawals and tracks realized taxable income."""
        # This function should be robust and use the logic from tax_planning.estimate_taxable_gap
        # but must also update the accounts in place.
        return 0.0, 0.0 # (ordinary_income, ltcg_income)

    def _apply_market_returns(self, accounts: Dict, market_return: float):
        """Applies market growth to all account balances."""
        for acct_name, acct_data in accounts.items():
            if 'balance' in acct_data:
                 acct_data['balance'] *= (1 + market_return)

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
