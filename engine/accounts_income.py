# engine/accounts_income.py

import numpy as np
import math
from typing import Tuple

from engine.rmd_tables import get_rmd_factor
from engine.def457b_tables import get_def457b_factor
from utils.tax_utils import SS_TAX_THRESHOLDS
from utils.ss_utils import get_full_retirement_age
from engine.withdrawal_engine import WithdrawalEngine

class AccountsIncomeEngine:
    def __init__(self, years, num_sims, inputs, annual_inflation, ages_person1, ages_person2, return_trajectories=True):

        self.years = years
        self.num_sims = num_sims
        self.inputs = inputs
        self.accounts = inputs.accounts

        self.withdrawal_engine = WithdrawalEngine(inputs, self.accounts)
        self.annual_inflation = annual_inflation
        
        self.ages_person1 = ages_person1
        self.ages_person2 = ages_person2
        
        self.return_trajectories = return_trajectories
       
    # ----------------------------------------------------------------------
    # Withdrawal Strategy delegation
    # ----------------------------------------------------------------------
    def _withdraw_from_hierarchy(self,
                                 cash_needed: float,
                                 accounts_bal: dict,
                                 simulate_only: bool) -> dict:
        """
        Delegates the withdrawal request to the dedicated WithdrawalEngine.
        """
        return self.withdrawal_engine._withdraw_from_hierarchy(
            cash_needed=cash_needed, 
            accounts_bal=accounts_bal,
            simulate_only=simulate_only
        )

    # ----------------------------------------------------------------------
    # Social Security — uses SS_TAX_THRESHOLDS from tax_utils.py (NOT indexed)
    # ----------------------------------------------------------------------
    def compute_taxable_ss(self, total_ss_benefit: float, other_agi: float, filing_status: str) -> float:
        """
        Exact replication of IRS Worksheet 1 logic using statutory (non-indexed) thresholds.
        Source: utils/tax_utils.py → SS_TAX_THRESHOLDS
        """
        # Normalize filing status key
        status_key = filing_status.replace("_filing_", "_").replace("jointly", "joint")
        if status_key not in SS_TAX_THRESHOLDS:
            status_key = "married_filing_jointly"  # safe fallback

        brackets = SS_TAX_THRESHOLDS[status_key]
        provisional_income = other_agi + 0.5 * total_ss_benefit

        taxable = 0.0
        for low, high, rate in brackets:
            if provisional_income <= low:
                break
            segment = min(provisional_income, high) - low
            taxable += segment * rate
            if provisional_income <= high:
                break

        return min(taxable, 0.85 * total_ss_benefit)
    
    # ----------------------------------------------------------------------
    # Social Security Benefit Calculation
    # ----------------------------------------------------------------------
    def compute_ss_benefit(self, current_year: int, inflation_index: float) -> float:
        """
        Calculates the total annual Social Security benefit for Person 1 and Person 2.
        Includes age-based adjustment factors, inflation indexing, and the Trust Fund 
        failure adjustment based on PlannerInputs.
        """
        
        # --- Helper: Calculates the SS benefit factor (percentage of FRA benefit) ---
        def _calculate_factor(current_age: int, start_age_years: int, start_age_months: int, fra_age_years: float) -> float:
            """Calculates the adjustment factor based on claiming age vs. FRA."""
            # Convert all ages to total years for comparison
            ss_start_age = start_age_years + (start_age_months / 12.0)
            
            # If the person has not reached their start age, the benefit is 0
            if current_age < ss_start_age:
                return 0.0
            
            # Determine the number of months difference between start age and FRA
            months_diff = int(round((ss_start_age - fra_age_years) * 12))
            
            if months_diff == 0:
                return 1.0 # Claiming at FRA
            
            if months_diff > 0:
                # Delayed Retirement Credit (DRC): 2/3 of 1% per month (8% per year)
                factor = 1.0 + (0.00667 * months_diff)
                return min(factor, 1.32) # Cap at age 70 (approx)
            else:
                # Early retirement reduction
                months_early = abs(months_diff)
                if months_early <= 36:
                    # First 36 months: 5/9 of 1% per month (~0.00556)
                    reduction = months_early * 0.00556
                else:
                    # Months after 36: 5/12 of 1% per month (~0.00417)
                    reduction = (36 * 0.00556) + ((months_early - 36) * 0.00417)
                
                factor = 1.0 - reduction
                return max(factor, 0.70) # Floor at age 62 (approx)

        # --- Main Function Logic ---
        year_index = current_year - self.inputs.current_year + 2 # year_index starts 2 years prior to current year
        total_ss_benefit = 0.0
        
        fra_amt_p1 = self.inputs.person1_ss_fra * 12 # convert monthly amount to annual ammount
        fra_amt_p2 = self.inputs.person2_ss_fra * 12 # convert monthly amount to annual ammount
        
        # 1. Person 1 Benefit
        age_p1 = self.ages_person1[year_index]
        if fra_amt_p1 > 0:
            p1_fra_age_calc = get_full_retirement_age(
                birth_year=self.inputs.person1_birth_year,
                birth_month=self.inputs.person1_birth_month
            )
            
            factor_p1 = _calculate_factor(
                current_age=age_p1,
                start_age_years=self.inputs.person1_ss_age_years,
                start_age_months=self.inputs.person1_ss_age_months,
                fra_age_years=p1_fra_age_calc
            )
            total_ss_benefit += fra_amt_p1 * factor_p1

        # 2. Person 2 Benefit
        if fra_amt_p2 > 0 and self.inputs.person2_birth_year is not None:
            age_p2 = self.ages_person2[year_index]
            p2_fra_age_calc = get_full_retirement_age(
                birth_year=self.inputs.person2_birth_year,
                birth_month=self.inputs.person2_birth_month
            )
            
            factor_p2 = _calculate_factor(
                current_age=age_p2,
                start_age_years=self.inputs.person2_ss_age_years,
                start_age_months=self.inputs.person2_ss_age_months,
                fra_age_years=p2_fra_age_calc
            )
            total_ss_benefit += fra_amt_p2 * factor_p2
        
        # 3. Apply Inflation
        ss_benefit_inflated = total_ss_benefit * inflation_index

        # 4. Apply Trust Fund Failure Logic
        if current_year >= self.inputs.ss_fail_year:
            # The benefit is reduced to (1 - ss_fail_percent) of the original
            ss_benefit_inflated *= (1 - self.inputs.ss_fail_percent)

        return ss_benefit_inflated

    # ----------------------------------------------------------------------
    # RMDs
    # ----------------------------------------------------------------------
    def compute_rmds(self,
                 accounts_bal: dict,
                 current_year: int) -> tuple[float, dict]:
        """
        Compute annual RMDs for the supplied account balances for `current_year`.

        Returns:
            total_rmds (float): sum of RMDs across accounts for the year
            rmd_by_account_annual (dict): mapping account_name -> annual RMD (0.0 if none)
        Notes / assumptions:
        - Uses account metadata fields: 'tax', 'owner', and for inherited accounts:
          'death_year', 'death_month' (optional), 'decedent_started_rmds' (bool or truthy string).
        - For owner accounts (tax == "traditional") uses SECURE 2.0 start-age logic
          integrated via get_rmd_factor().
        - For inherited accounts:
           * If decedent_started_rmds == True -> treat as "inherited continuing"
             and compute annual RMD using get_rmd_factor(..., inherited_continuing=True).
           * Else if death_year >= 2020 -> apply the 10-year rule => **no** annual
             RMDs enforced here (beneficiary may withdraw within 10 years). This is
             consistent with the common interpretation of the post-SECURE rules.
           * Else (death_year < 2020 and decedent had not started RMDs) -> conservative
             fallback: compute an annual draw using get_rmd_factor(..., inherited_continuing=False).
        - All ages are computed on a year basis (age = current_year - birth_year).
          If you want month-level precision use birth_month & current month; add that and
          I will incorporate month arithmetic.
        """
        from engine.rmd_tables import get_rmd_factor

        total_rmds = 0.0
        rmd_by_account_annual = {}

        # helper to parse truthy booleans from XML (accept "True"/"true"/True)
        def _truthy(val):
            if isinstance(val, bool):
                return val
            if val is None:
                return False
            s = str(val).strip().lower()
            return s in ("true", "t", "1", "yes", "y")

        # small helper to get owner birth info from self.inputs
        def _owner_birth(owner_key: str):
            if owner_key == "person1":
                return self.inputs.person1_birth_year, getattr(self.inputs, "person1_birth_month", None)
            elif owner_key == "person2":
                return self.inputs.person2_birth_year, getattr(self.inputs, "person2_birth_month", None)
            else:
                return None, None

        for acct_name, acct_meta in self.accounts.items():
            # default zero
            rmd_by_account_annual[acct_name] = 0.0

            acct_balance = accounts_bal.get(acct_name, {}).get("balance", 0.0)
            acct_tax = acct_meta.get("tax", "").lower()

            # skip non-RMD accounts and zero balances
            if acct_balance is None or acct_balance <= 0 or acct_tax not in ("traditional", "inherited"):
                continue

            # OWNER-LOCATED (owner is either person1 or person2)
            owner = acct_meta.get("owner", None)
            owner_birth_year, owner_birth_month = _owner_birth(owner)

            # Helper compute integer age at calendar year (year resolution)
            def _age_at_year(birth_year, birth_month=None):
                if birth_year is None:
                    return None
                # conservative: use year-only age (e.g., 2025 - 1955 = 70)
                return current_year - birth_year

            # -------------------------
            # Traditional (owner) accounts
            # -------------------------
            if acct_tax == "traditional":
                owner_age = _age_at_year(owner_birth_year, owner_birth_month)
                if owner_age is None:
                    # cannot determine age -> skip (safest is zero)
                    rmd_by_account_annual[acct_name] = 0.0
                    continue

                # request factor from table; get_rmd_factor returns 0.0 if age < start
                rmd_factor = get_rmd_factor(age=owner_age, birth_year=owner_birth_year,
                                            inherited_continuing=False, use_pre_2022_table=False)
                if rmd_factor > 0:
                    rmd_amount = acct_balance / rmd_factor
                    rmd_by_account_annual[acct_name] = rmd_amount
                    total_rmds += rmd_amount
                else:
                    rmd_by_account_annual[acct_name] = 0.0

            # -------------------------
            # Inherited accounts
            # -------------------------
            elif acct_tax == "inherited":
                # required metadata: death_year (and optionally death_month), decedent_started_rmds
                death_year = acct_meta.get("death_year", None)
                death_month = acct_meta.get("death_month", None)
                dec_started = _truthy(acct_meta.get("decedent_started_rmds", False))

                # beneficiary = acct_meta['owner'] (owner is beneficiary)
                beneficiary_birth_year, beneficiary_birth_month = _owner_birth(owner)
                beneficiary_age = _age_at_year(beneficiary_birth_year, beneficiary_birth_month)
                if beneficiary_age is None:
                    # cannot compute beneficiary age -> skip
                    rmd_by_account_annual[acct_name] = 0.0
                    continue

                # If decedent had already started RMDs -> continue using "inherited_continuing"
                if dec_started:
                    # Use inherited_continuing True to pick pre-2022 table inside get_rmd_factor
                    rmd_factor = get_rmd_factor(age=beneficiary_age,
                                                birth_year=beneficiary_birth_year,
                                                inherited_continuing=True,
                                                use_pre_2022_table=False)
                    if rmd_factor > 0:
                        rmd_amount = acct_balance / rmd_factor
                        rmd_by_account_annual[acct_name] = rmd_amount
                        total_rmds += rmd_amount
                    else:
                        rmd_by_account_annual[acct_name] = 0.0
                    continue

                # If decedent did NOT start RMDs:
                # Apply 10-year rule for deaths on/after 2020 -> typically no annual RMD required,
                # beneficiary may distribute any time in 10 years. We implement the common rule:
                #  -> No annual RMD enforced here (rmd = 0.0). If you want equal annualization,
                #     we can implement that as an option.
                try:
                    death_year_int = int(death_year) if death_year is not None else None
                except Exception:
                    death_year_int = None

                if death_year_int is not None and death_year_int >= 2020:
                    # 10-year rule --> no required annual RMD enforced by this function
                    rmd_by_account_annual[acct_name] = 0.0
                    continue

                # For older deaths (pre-2020) where decedent had not started RMDs,
                # it's common to require beneficiary to take RMDs over life expectancy.
                # Use beneficiary age and pre-2022 table as a conservative fallback.
                rmd_factor = get_rmd_factor(age=beneficiary_age,
                                            birth_year=beneficiary_birth_year,
                                            inherited_continuing=False,
                                            use_pre_2022_table=True)
                if rmd_factor > 0:
                    rmd_amount = acct_balance / rmd_factor
                    rmd_by_account_annual[acct_name] = rmd_amount
                    total_rmds += rmd_amount
                else:
                    rmd_by_account_annual[acct_name] = 0.0

        return total_rmds, rmd_by_account_annual

    # ----------------------------------------------------------------------
    # Trust Income 
    # ----------------------------------------------------------------------
    def compute_trust_income(self, accounts: dict, current_year: int, inflation_index: float) -> float:
        """
        Calculates the total annual taxable income generated by Trust accounts.
        The balance comes from the dynamic simulation state (accounts), but the fixed
        yield parameter comes from the engine's static metadata (self.accounts).
        """
        total = 0.0
        
        # Iterate over the dynamic accounts (to get the current balance)
        for acct_name, acct_state in accounts.items(): 
            
            # Look up the static metadata for the account type and yield
            # self.accounts holds the original inputs.accounts (with the correct float)
            acct_metadata = self.accounts.get(acct_name, {})
            
            # Check if this account is a Trust account based on static metadata
            if acct_metadata.get("tax") == "trust":
                
                # Get the dynamic balance for the current year
                balance = acct_state.get("balance", 0.0)
                
                # Get the static yield rate (e.g., 0.015) from the metadata
                yield_rate = acct_metadata.get("income", 0.0)
                
                # If yield rate is still 0 (e.g., account wasn't found or yield is missing), skip
                if yield_rate > 0.0:
                    trust_income_amount = balance * yield_rate
                    total += trust_income_amount
                    
        return total
    # ----------------------------------------------------------------------
    # Def 457b 
    # ----------------------------------------------------------------------
    def compute_def457b_income(self, accounts: dict, current_year: int) -> float:
        """
        Calculates the total annual 457b income across all accounts, using
        centralized drawdown parameters for Person 1 and Person 2.
        """
        total_income = 0.0

        # --- 1. Calculate Divisor for Person 1 ---
        p1_draw_age_years = self.inputs.person1_def457b_age_years
        p1_draw_age_months = self.inputs.person1_def457b_age_months
        
        # Convert total drawdown months to years (required by get_def457b_factor)
        p1_drawdown_years = self.inputs.person1_def457b_drawdown_months / 12.0 
        #
        #***EVENTUALLY WANT TO CONVERT TO QUARTER SO THIS COULD START MID-YEAR
        #
        p1_start_year = math.ceil(self.inputs.person1_birth_year + p1_draw_age_years + (self.inputs.person1_birth_month + p1_draw_age_months) / 12.0)
        
        p1_multiplier = 0.0

        if current_year >= p1_start_year and p1_drawdown_years > 0:
            p1_multiplier = get_def457b_factor(current_year, p1_start_year, p1_drawdown_years)

        #print(f"457b Start year {p1_start_year}  current year {current_year}   Drawdown over {p1_drawdown_years} years   Multiplier = {p1_multiplier}")

        # --- 2. Calculate Divisor for Person 2 ---
        p2_draw_age_years = self.inputs.person2_def457b_age_years
        p2_draw_age_months = self.inputs.person2_def457b_age_months
        
        # Convert total drawdown months to years
        p2_drawdown_years = self.inputs.person2_def457b_drawdown_months / 12.0
        #
        #***EVENTUALLY WANT TO CONVERT TO QUARTER SO THIS COULD START MID-YEAR
        #
        p2_start_year = math.ceil(self.inputs.person2_birth_year + p2_draw_age_years + (self.inputs.person2_birth_month + p2_draw_age_months) / 12.0)
        p2_multiplier = 0.0

        if current_year >= p2_start_year and p2_drawdown_years > 0:
             p2_multiplier = get_def457b_factor(current_year, p2_start_year, p2_drawdown_years)

        # --- 3. Iterate over all 457b accounts and calculate withdrawal ---
        # We iterate over the account METADATA (self.accounts) to get the owner
        for acct_name, acct_metadata in self.accounts.items():
            if acct_metadata.get("tax") != "def457b":
                continue

            owner = acct_metadata.get("owner")
            
            # Get the current numerical balance from the simulation state ('accounts')
            acct_state = accounts.get(acct_name, {})
            balance = acct_state.get("balance", 0.0)
            
            def457b_amount = 0.0
            
            if balance > 0.0:
                if owner == "person1" and p1_multiplier > 0:
                    def457b_amount = balance * p1_multiplier
                elif owner == "person2" and p2_multiplier > 0:
                    def457b_amount = balance * p2_multiplier
            
            total_income += def457b_amount

        return total_income

    # ----------------------------------------------------------------------
    # Pension (unchanged, uses inputs directly)
    # ----------------------------------------------------------------------
    def compute_pension_income(self, year: int, inflation_index: float) -> float:
        amount1 = 0.0
        amount2 = 0.0
        
        age_p1 = self.ages_person1[year - self.inputs.current_year]
        pension_age_1 = (self.inputs.person1_pension_age_years +
                       self.inputs.person1_pension_age_months / 12)

        if age_p1 >= pension_age_1:
            amount1 = self.inputs.person1_pension_amount * 12 # need to convert monthly to annual
            if self.inputs.person1_pension_cola:
                amount1 *= inflation_index
        
        age_p2 = self.ages_person2[year - self.inputs.current_year]
        pension_age_2 = (self.inputs.person2_pension_age_years +
                       self.inputs.person2_pension_age_months / 12)

        if age_p2 >= pension_age_2:
            amount2 = self.inputs.person2_pension_amount * 12 # need to convert monthly to annual
            if self.inputs.person2_pension_cola:
                amount2 *= inflation_index

        amount = amount1 + amount2
        return amount

    # ----------------------------------------------------------------------
    # Pension 
    # ----------------------------------------------------------------------
    def compute_pension_income(self, year: int, inflation_index: float) -> float:
        amount1 = 0.0
        amount2 = 0.0
        
        age_p1 = self.ages_person1[year - self.inputs.current_year]
        pension_age_1 = (self.inputs.person1_pension_age_years +
                       self.inputs.person1_pension_age_months / 12)

        if age_p1 >= pension_age_1:
            amount1 = self.inputs.person1_pension_amount * 12 # need to convert monthly to annual
            if self.inputs.person1_pension_cola:
                amount1 *= inflation_index
        
        age_p2 = self.ages_person2[year - self.inputs.current_year]
        pension_age_2 = (self.inputs.person2_pension_age_years +
                       self.inputs.person2_pension_age_months / 12)

        if age_p2 >= pension_age_2:
            amount2 = self.inputs.person2_pension_amount * 12 # need to convert monthly to annual
            if self.inputs.person2_pension_cola:
                amount2 *= inflation_index

        amount = amount1 + amount2
        return amount

    # ----------------------------------------------------------------------
    # Salary
    # ----------------------------------------------------------------------
    def compute_salary_income(self, year: int, inflation_index: float) -> float:
        amount1 = 0.0
        amount2 = 0.0
        
        age_p1 = self.ages_person1[year - self.inputs.current_year]
        retire_age_1 = (self.inputs.person1_ret_age_years +
                       self.inputs.person1_ret_age_months / 12)

        if age_p1 >= retire_age_1:
            amount1 = self.inputs.person1_salary_amount * 12 # need to convert monthly to annual
            amount1 *= inflation_index
        
        age_p2 = self.ages_person2[year - self.inputs.current_year]
        retire_age_2 = (self.inputs.person2_ret_age_years +
                       self.inputs.person2_ret_age_months / 12)

        if age_p2 >= retire_age_2:
            amount2 = self.inputs.person2_salary_amount * 12 # need to convert monthly to annual
            amount2 *= inflation_index

        amount = amount1 + amount2
        return amount

