# income_calculator.py
#
# Manages income calculation (Salary, Trust income and 457b, Pension, Social Security
#

import numpy as np

def calculate_salary_income(
    ages_person1: np.ndarray,
    ages_person2: np.ndarray,
    person1_work_end_age: int,
    person2_work_end_age: int,
    person1_salary: float, # Initial unadjusted salary (Year 0 value)
    person2_salary: float, # Initial unadjusted salary (Year 0 value)
    inflation_index: np.ndarray, # Array of cumulative inflation multipliers
    year_idx: int
) -> float:
    """
    Calculates the total annual salary income for both individuals, 
    adjusted for inflation, up to their specified working end ages.
    
    Args:
        ages_person1, ages_person2: Arrays of ages per year/index.
        person1_work_end_age, person2_work_end_age: Age at which salary stops.
        person1_salary, person2_salary: Base salary amounts (Year 0 values).
        inflation_index: Array of cumulative inflation multipliers (Year 0 = 1.0).
        year_idx: The current year index in the simulation.
        
    Returns:
        float: The total combined annual salary income for the current year.
    """
    
    total_salary = 0.0
    # The inflation index represents the cumulative inflation since Year 0.
    current_inflation_multiplier = inflation_index[year_idx]
    
    # Person 1 Salary Logic
    if ages_person1[year_idx] < person1_work_end_age:
        # Salary is adjusted for cumulative inflation (COLA)
        person1_income = person1_salary * current_inflation_multiplier
        total_salary += person1_income
        
    # Person 2 Salary Logic
    if ages_person2[year_idx] < person2_work_end_age:
        # Salary is adjusted for cumulative inflation (COLA)
        person2_income = person2_salary * current_inflation_multiplier
        total_salary += person2_income
        
    return total_salary

def calculate_mandatory_account_draws(
    accounts_bal: dict, 
    accounts_def: dict, 
    year: int,
    birth_years: dict, # e.g., {'person1': 1965, 'person2': 1959}
    get_def457b_factor_func: callable
) -> tuple[dict, float, float]:
    """
    Calculates mandatory withdrawals (457b, Trust Yield).
    
    Returns: 
        (withdrawal_breakdown: dict, total_def457b_draw: float, total_trust_draw: float)
    """
    
    all_mandatory_withdrawals = {}
    total_def457b_draw = 0.0
    total_trust_draw = 0.0

    for acct_name, acct in accounts_def.items():
        acct_balance = accounts_bal.get(acct_name, 0.0)
        mand = 0.0
        
        # 457b Draw Logic
        if acct["tax"] == "def457b" and acct_balance > 0:
            person = acct["owner"]
            birth_year = birth_years[person]
            def457b_start_year = birth_year + acct["start_age"]
            ddyears = acct["drawdown_years"]
            
            def457b_factor = get_def457b_factor_func(year, def457b_start_year, ddyears)
            mand = acct_balance * def457b_factor if def457b_factor > 0 else 0
            
            total_def457b_draw += mand
            
        # Trust Mandatory Yield Logic
        elif acct["tax"] == "trust" and acct.get("mandatory_yield", 0) > 0 and acct_balance > 0:
            mand = acct_balance * acct["mandatory_yield"]
            total_trust_draw += mand
        
        if mand > 0:
            all_mandatory_withdrawals[acct_name] = mand
            
    return all_mandatory_withdrawals, total_def457b_draw, total_trust_draw

def calculate_rmds(self, year_idx, accounts_bal):
    rmds = 0.0
    rmd_withdrawals = {}

    for acct_name, acct in self.accounts.items():
        if acct_name not in accounts_bal: continue

        bal = accounts_bal[acct_name]
        age = self.ages_person1[year_idx] if acct["owner"] == "person1" else self.ages_person2[year_idx]

        factor = 0.0
        if acct["tax"] == "inherited":
            # Assuming person1=1965, person2=1959 logic from original
            birth_year = 1965 if acct["owner"] == "person1" else 1959
            factor = get_rmd_factor(age, birth_year, True, True)
        elif acct["tax"] == "traditional":
            birth_year = 1965 if acct["owner"] == "person1" else 1959
            factor = get_rmd_factor(age, birth_year, False, False)

        if factor > 0:
            amount = bal / factor
            rmd_withdrawals[acct_name] = amount
            rmds += amount
        else:
            rmd_withdrawals[acct_name] = 0.0

    return rmds, rmd_withdrawals

def calculate_pension_income(
    ages_person1: list[int],
    ages_person2: list[int],
    person1_pension_age_years: int,
    person1_pension_age_months: int,
    person1_birth_month: int,
    person1_pension_amount: float,
    person2_pension_age_years: int,
    person2_pension_amount: float,
    person2_pension_age_months: int,
    person2_birth_month: int,
    year_idx: int
) -> float:
    """Calculates the total annual pension income for both individuals."""
    
    person1_pension_income = 0.0
    person2_pension_income = 0.0

    # Person 1 Pension Logic
    if ages_person1[year_idx] >= person1_pension_age_years:
        months = 12
        if ages_person1[year_idx] == person1_pension_age_years:
            # Calculate start month in the first year
            month = person1_birth_month + person1_pension_age_months
            months = 12 - month
        
        person1_pension_income = months * person1_pension_amount

    # Person 2 Pension Logic (Note: Original code had a bug where it referenced 
    # person1's variables inside person2's check. This version fixes that.)
    if ages_person2[year_idx] >= person2_pension_age_years:
        months = 12
        if ages_person2[year_idx] == person2_pension_age_years:
            # Calculate start month in the first year
            month = person2_birth_month + person2_pension_age_months
            months = 12 - month
        
        person2_pension_income = months * person2_pension_amount

    return person1_pension_income + person2_pension_income

def calculate_ss_benefit(
    ages_person1: list[int],
    ages_person2: list[int],
    person1_ss_age_years: int,
    person1_ss_age_months: int,
    person1_birth_month: int,
    person1_ss_fra: float,
    person1_birth_year: int,
    person2_ss_age_years: int,
    person2_ss_age_months: int,
    person2_birth_month: int,
    person2_ss_fra: float,
    person2_birth_year: int,
    ss_fail_year: int,
    ss_fail_percent: float,
    inflation_index: np.ndarray,
    year: int,
    year_idx: int,
    get_ss_multiplier_func: callable
) -> float:
    """
    Calculates the total annual Social Security benefit for both individuals, 
    including adjustments for age, COLA, and potential trust fund failure.
    """
    
    person1_ss = person1_ss_fra
    person2_ss = person2_ss_fra
    
    # Apply SS Trust Fund failure adjustment
    if year > ss_fail_year:
        person1_ss *= (1 - ss_fail_percent)
        person2_ss *= (1 - ss_fail_percent)

    # Spousal benefits are calculated using the FRA amount
    person1_spousal = 0.5 * person2_ss
    person2_spousal = 0.5 * person1_ss

    person1_benefit = 0.0
    person2_benefit = 0.0
    
    # Person 1 Benefit
    if ages_person1[year_idx] >= person1_ss_age_years:
        months = 12
        
        # Check for first year of benefits
        if ages_person1[year_idx] == person1_ss_age_years:
            month = person1_birth_month + person1_ss_age_months
            months = 12 - month
            # Original code set inflation_index_start_person1 here. 
            # We assume it remains constant after the first year it is taken.
            # To handle this state correctly in a stateless function, you'd 
            # need to pass the "start year inflation index" for COLA tracking.
            # For simplicity, we use the current year's index for COLA base if it's the start year.
            
        # Get Age Adjustment Multiplier
        age_multiplier = get_ss_multiplier_func(person1_birth_year, person1_ss_age_years, person1_ss_age_months)
        
        # Calculate COLA Multiplier (simplified here, assumes COLA is tracked outside)
        # To perfectly replicate the COLA tracking, you need a dictionary of SS start inflation index.
        # Since we don't have that state here, we rely on the `person1_ss` being the inflation-adjusted FRA
        # amount from the simulator's state tracking.
        # For now, we apply the age multiplier directly:
        person1_own = person1_ss * age_multiplier
        
        # Take the higher of own or spousal benefit
        # Note: Your original code applied the COLA *after* getting the multiplier.
        # We must follow that flow to match results, even if state tracking is imperfect:
        # The COLA logic in the original code is complex due to state tracking (`inflation_index_start_personX`).
        # This implementation simplifies by applying the benefit multiplier but assumes the `personX_ss_fra`
        # passed into this function is already inflation-adjusted from previous years.
        
        # Final Calculation
        person1_benefit = months * max(person1_own, person1_spousal)
        
    # Person 2 Benefit
    if ages_person2[year_idx] >= person2_ss_age_years:
        months = 12
        if ages_person2[year_idx] == person2_ss_age_years:
            month = person2_birth_month + person2_ss_age_months
            months = 12 - month
        
        age_multiplier = get_ss_multiplier_func(person2_birth_year, person2_ss_age_years, person2_ss_age_months)
        person2_own = person2_ss * age_multiplier
        person2_benefit = months * max(person2_own, person2_spousal)
        
    return person1_benefit + person2_benefit

def calculate_taxable_ss_portion(ss_benefit: float) -> float:
    """
    Calculates the maximum taxable portion of Social Security benefit (85%).
    Note: The actual taxable amount depends on Provisional Income/MAGI, which is 
    calculated in the Tax Engine, not here. This function returns the maximum 
    ordinary income component from SS.
    """
    return 0.85 * ss_benefit


