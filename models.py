# models.py 
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class PlannerInputs:
    # Core
    current_year: int
    end_age: int
    filing_status: str
    state_of_residence: str

    # Person 1
    person1_birth_year: int
    person1_birth_month: int
    
    person1_ret_age_years: int
    person1_ret_age_months: int
    person1_salary_amount: float
    
    person1_ss_age_years: int
    person1_ss_age_months: int
    person1_ss_fra: float
    
    person1_pension_age_years: int
    person1_pension_age_months: int
    person1_pension_amount: float
    person1_pension_cola: bool
    
    person1_def457b_age_years: int
    person1_def457b_age_months: int
    person1_def457b_drawdown_months: int

    # Person 2
    person2_birth_year: int
    person2_birth_month: int
    
    person2_ret_age_years: int
    person2_ret_age_months: int
    person2_salary_amount: float
    
    person2_ss_age_years: int
    person2_ss_age_months: int
    person2_ss_fra: float
    
    person2_pension_age_years: int
    person2_pension_age_months: int    
    person2_pension_amount: float
    person2_pension_cola: bool
    
    person2_def457b_age_years: int
    person2_def457b_age_months: int
    person2_def457b_drawdown_months: int

    # Portfolio & strategy
    accounts: Dict[str, Any]
    #portfolio_data: Dict[str, Any]

    success_threshold: float
    avoid_ruin_threshold: float
    
    nsims: int
    tax_strategy: str
    roth_tax_bracket: str
    roth_irmaa_threshold: str
    max_roth: float
    base_annual_spending: float
    withdrawal_rate: float
    travel: float
    gifting: float
    magi_1: float
    magi_2: float
    ss_fail_year: int
    ss_fail_percent: float

    return_trajectories: bool = True
