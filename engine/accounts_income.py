# accounts_income.py
import numpy as np

def compute_monthly_account_income(year, month, accounts, accounts_bal, rmds, ss_benefit, inflation_this_year, month_idx):
    """
    Compute income from def457b, trust, RMDs, pensions etc. at month granularity.
    Returns dicts with monthly income per account type and totals.
    """
    def457b_withdrawal = {}
    def457b_total = 0.0
    trust_withdrawal = {}
    trust_income_total = 0.0

    for acct_name, acct in accounts.items():
        acct_balance = accounts_bal[acct_name]

        # 457b
        if acct["tax"] == "def457b":
            person = acct["owner"]
            byp = f"{person}_birth_year"
            start_year = getattr(acct, byp) + acct["start_age"]  # could be fractional year
            drawdown_years = acct["drawdown_years"]
            factor = get_def457b_factor(year, start_year, drawdown_years)  # returns annual factor
            mand = acct_balance * factor if factor > 0 else 0
            def457b_withdrawal[acct_name] = mand / 12  # monthly
            def457b_total += mand / 12

        # Trust income (mandatory yield)
        if acct["tax"] == "trust" and acct.get("mandatory_yield", 0) > 0:
            mand = acct_balance * acct["mandatory_yield"]
            trust_withdrawal[acct_name] = mand / 12
            trust_income_total += mand / 12

    # base income
    monthly_income = {
        "def457b": def457b_total,
        "trust": trust_income_total,
        "rmds": rmds / 12,
        "ss": ss_benefit / 12,
    }
    total_income = sum(monthly_income.values())

    return monthly_income, total_income, def457b_withdrawal, trust_withdrawal
