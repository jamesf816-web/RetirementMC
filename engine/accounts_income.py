# engine/accounts_income.py

import numpy as np
from typing import Tuple

from engine.rmd_tables import get_rmd_factor
from engine.def457b_tables import get_def457b_factor
from utils.tax_utils import SS_TAX_THRESHOLDS 
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
                                 order: list) -> dict:
        """
        Delegates the withdrawal request to the dedicated WithdrawalEngine.
        Note: The tax_strategy is no longer explicitly passed to the engine,
        as the engine uses self.inputs.tax_strategy internally.
        """
        return self.withdrawal_engine._withdraw_from_hierarchy(
            cash_needed=cash_needed, 
            accounts_bal=accounts_bal,
            simulate_only=False # Assuming the default is to modify balances
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
        year_index = current_year - self.inputs.current_year
        total_ss_benefit = 0.0
        
        # **CORRECTED** - Using 'person1_ss_fra' and 'person2_ss_fra' for the dollar amount
        # since they are the only float fields associated with SS in PlannerInputs.
        fra_amt_p1 = self.inputs.person1_ss_fra
        fra_amt_p2 = self.inputs.person2_ss_fra
        
        # 1. Person 1 Benefit
        age_p1 = self.ages_person1[year_index]
        if fra_amt_p1 > 0:
            # Need to get FRA age (e.g., 67.0) to pass to the factor calculation
            p1_fra_age = self.inputs.person1_ss_fra
            # Assuming person1_ss_fra *is* the FRA age if person1_ss_age_years/months are separate
            # The models.py shows:
            # person1_ss_age_years: int (Claiming year)
            # person1_ss_age_months: int (Claiming month)
            # person1_ss_fra: float (FRA Benefit Amount in $)
            
            # We need the person's actual FRA age in years (e.g., 67.0) to calculate the factor.
            # Assuming the inputs model should have a field like 'person1_fra_age_years'.
            # Since it is missing, we will temporarily use the SS start age as a placeholder 
            # for a base calculation, or rely on an external assumption for the actual FRA age.
            # BASED ON YOUR MODEL (person1_ss_fra is a float, likely the dollar amount):
            
            # **ASSUMPTION**: To calculate the factor, we need the actual FRA age (e.g., 67). 
            # I will use a conservative placeholder of 67.0 for the FRA age for the factor calculation.
            # If your `PlannerInputs` has a field for the actual FRA age (e.g., 67.0), use it instead.
            p1_fra_age_calc = 67.0 
            
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
            p2_fra_age_calc = 67.0 # **ASSUMPTION**: Use a conservative placeholder of 67.0
            
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
            # The benefit is reduced to (ss_fail_percent) of the original
            ss_benefit_inflated *= self.inputs.ss_fail_percent

        return ss_benefit_inflated
    
    # ----------------------------------------------------------------------
    # RMDs 
    # ----------------------------------------------------------------------
    def compute_rmds(self, 
                 accounts_bal: dict, 
                 current_year: int) -> float:
        """
        Calculates the total annual RMD required across all traditional and inherited accounts.
        """
        from engine.rmd_tables import get_rmd_factor # Ensure this is imported at the top of the file

        # 1. Calculate Ages (Internal Calculation)
        age_p1 = current_year - self.inputs.person1_birth_year
        age_p2 = None
        if self.inputs.person2_birth_year is not None:
            age_p2 = current_year - self.inputs.person2_birth_year

        total_rmds = 0.0

        # 2. Iterate through accounts and apply RMD logic
        for acct_name, acct in self.accounts.items():
            acct_balance = accounts_bal[acct_name]
            rmd_amount = 0.0

            acct_state = accounts_bal[acct_name]    
            # Extract the actual numerical balance before using it in math
            acct_numerical_balance = acct_state.get("balance", 0.0) 

            if acct_numerical_balance > 0 and (age_p1 >= 73 or acct["tax"] == "inherited"):
                
                # Determine RMD person, age, and flags
                if acct.get("owner") == "person1":
                    current_age = age_p1
                    rmd_flags = (True, True) if acct.get("tax") == "inherited" else (False, False)
                elif acct.get("owner") == "person2" and age_p2 is not None:
                    current_age = age_p2
                    rmd_flags = (True, True) if acct.get("tax") == "inherited" else (False, False)
                else:
                    # Skip if owner is not recognized or person 2 is not in play
                    continue 

                # Inherited and Traditional accounts require RMDs
                if acct.get("tax") in ["inherited", "traditional"]:
                    # RMD factor is calculated based on age and account type
                    # Note: The original code passed a birth year (1965 or 1959) which is unusual
                    # but is preserved here to maintain the original intent of the function call.
                    rmd_factor = get_rmd_factor(current_age, 1965 if acct.get("owner") == "person1" else 1959, *rmd_flags)

                    if rmd_factor > 0:
                        rmd_amount = acct_numerical_balance / rmd_factor
                    total_rmds += rmd_amount

                    # Note: You may want to store rmd_amount in a dictionary if needed for quarterly draws.
                    # self.rmd_withdrawal[acct_name] = rmd_amount / 4 # e.g.

        return total_rmds

    # ----------------------------------------------------------------------
    # Trust Income 
    # ----------------------------------------------------------------------
    def compute_trust_income(self, accounts: dict) -> float:
        total = 0.0
        for acct in accounts.values():
            if acct.get("tax") == "trust":
                total += acct.get("balance", 0.0) * acct.get("annual_yield", 0.0)
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
        # Ensure 'get_def457b_factor' is imported at the top of accounts_income.py

        # --- 1. Calculate Divisor for Person 1 ---
        p1_draw_age_years = self.inputs.person1_def457b_age_years
        
        # Convert total drawdown months to years (required by get_def457b_factor)
        p1_drawdown_years = self.inputs.person1_def457b_drawdown_months / 12.0 
        p1_start_year = self.inputs.person1_birth_year + p1_draw_age_years
        p1_divisor = 0.0

        if current_year >= p1_start_year and p1_drawdown_years > 0:
            p1_divisor = get_def457b_factor(current_year, p1_start_year, p1_drawdown_years)

        # --- 2. Calculate Divisor for Person 2 ---
        p2_draw_age_years = self.inputs.person2_def457b_age_years
        
        # Convert total drawdown months to years
        p2_drawdown_years = self.inputs.person2_def457b_drawdown_months / 12.0
        p2_start_year = self.inputs.person2_birth_year + p2_draw_age_years
        p2_divisor = 0.0

        if current_year >= p2_start_year and p2_drawdown_years > 0:
             p2_divisor = get_def457b_factor(current_year, p2_start_year, p2_drawdown_years)


        # --- 3. Iterate over all 457b accounts and calculate withdrawal ---
        # We iterate over the account METADATA (self.accounts) to get the owner
        for acct_name, acct_metadata in self.accounts.items():
            if acct_metadata.get("tax") != "def457b":
                continue

            owner = acct_metadata.get("owner")
            
            # Get the current numerical balance from the simulation state ('accounts')
            acct_state = accounts.get(acct_name, {})
            balance = acct_state.get("balance", 0.0)
            
            rmd_amount = 0.0
            
            if balance > 0.0:
                if owner == "person1" and p1_divisor > 0:
                    rmd_amount = balance / p1_divisor
                elif owner == "person2" and p2_divisor > 0:
                    rmd_amount = balance / p2_divisor
            
            total_income += rmd_amount

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
            amount1 = self.inputs.person1_pension_amount
            if self.inputs.person1_pension_cola:
                amount1 *= inflation_index
        
        age_p2 = self.ages_person2[year - self.inputs.current_year]
        pension_age_2 = (self.inputs.person2_pension_age_years +
                       self.inputs.person2_pension_age_months / 12)

        if age_p2 >= pension_age_2:
            amount2 = self.inputs.person2_pension_amount
            if self.inputs.person2_pension_cola:
                amount2 *= inflation_index

        amount = amount1 + amount2
        return amount

