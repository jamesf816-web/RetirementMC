import numpy as np
import math

class AccountsIncomeEngine:
def **init**(self, years, num_sims, annual_inflation, ages_person1, ages_person2,
gifting, estimated_taxes, medicare_parts, return_trajectories=True):
self.years = years
self.num_sims = num_sims
self.annual_inflation = annual_inflation
self.ages_person1 = ages_person1
self.ages_person2 = ages_person2
self.gifting = gifting
self.estimated_taxes = estimated_taxes
self.medicare_parts = medicare_parts
self.return_trajectories = return_trajectories

```
# -------------------------------
# Monthly age update
# -------------------------------
def update_monthly_ages(self, year_idx, month):
    self.current_age_person1 = self.ages_person1[year_idx] + (month - 1)/12
    self.current_age_person2 = self.ages_person2[year_idx] + (month - 1)/12

# -------------------------------
# RMDs (quarterly)
# -------------------------------
def compute_rmds(self, accounts, accounts_bal, year_idx):
    rmds = {}
    for acct_name, acct in accounts.items():
        if acct["tax"] in ["inherited", "traditional"]:
            balance = accounts_bal[acct_name]
            rmds[acct_name] = max(0, balance / 27.4)  # Example life expectancy divisor
    return rmds

# -------------------------------
# Pension income (yearly, monthly scale handled in loop)
# -------------------------------
def get_pension_income(self, year_idx):
    return 30000  # Placeholder: replace with your real pension logic

# -------------------------------
# Social Security benefit (yearly)
# -------------------------------
def get_ss_benefit(self, year_idx):
    return 24000  # Placeholder

# -------------------------------
# Quarterly withdrawals
# -------------------------------
def handle_quarterly_withdrawals(self, accounts, accounts_bal, rmds, year_idx, quarter_index):
    # Withdraw RMDs and inherited/traditional
    for acct_type in ["inherited", "traditional"]:
        for acct_name, acct in accounts.items():
            if acct["tax"] != acct_type:
                continue
            withdraw_amt = min(accounts_bal[acct_name], rmds.get(acct_name,0)/4)
            accounts_bal[acct_name] -= withdraw_amt

    # Withdraw 457b (quarterly)
    for acct_type in ["def457b"]:
        for acct_name, acct in accounts.items():
            if acct["tax"] != acct_type:
                continue
            withdraw_amt = min(accounts_bal[acct_name], 15000/4)  # Example annual 457b draw
            accounts_bal[acct_name] -= withdraw_amt

    # Withdraw Trust Income
    for acct_type in ["trust"]:
        for acct_name, acct in accounts.items():
            if acct["tax"] != acct_type:
                continue
            withdraw_amt = 50000/4  # Quarterly Trust draw
            accounts_bal[acct_name] -= withdraw_amt

# -------------------------------
# Portfolio draw for essentials and taxes
# -------------------------------
def compute_portfolio_draws(self, accounts, accounts_bal, year_idx, quarter_index):
    portfolio_draw_add = 20000  # Replace with calculation: essentials + travel + estimated tax + medicare
    # Example: withdraw in order: taxable, trust, inherited, traditional, roth
    withdrawal_order = ["taxable", "trust", "inherited", "traditional", "roth"]
    remaining = portfolio_draw_add
    for acct_type in withdrawal_order:
        for acct_name, acct in accounts.items():
            if acct["tax"] != acct_type or remaining <= 0:
                continue
            withdraw_amt = min(accounts_bal[acct_name], remaining)
            accounts_bal[acct_name] -= withdraw_amt
            remaining -= withdraw_amt
            if remaining <= 0:
                break
        if remaining <= 0:
            break
    return portfolio_draw_add

# -------------------------------
# Travel (lumpy)
# -------------------------------
def handle_travel(self, accounts, accounts_bal, year_idx, quarter_index, quarter_label):
    travel_cost = 10000 if quarter_label=="Q1" else 15000
    remaining = travel_cost
    for acct_type in ["taxable", "trust", "inherited", "traditional", "roth"]:
        for acct_name, acct in accounts.items():
            if acct["tax"] != acct_type or remaining <= 0:
                continue
            withdraw_amt = min(accounts_bal[acct_name], remaining)
            accounts_bal[acct_name] -= withdraw_amt
            remaining -= withdraw_amt
            if remaining <= 0:
                break
        if remaining <= 0:
            break

# -------------------------------
# Roth conversions (Q4)
# -------------------------------
def handle_roth_conversions(self, accounts, accounts_bal, year_idx):
    conv_amount = 10000
    for person in ["person1","person2"]:
        trad_accts = {k:v for k,v in accounts.items() if v["owner"]==person and v["tax"]=="traditional"}
        roth_accts = {k:v for k,v in accounts.items() if v["owner"]==person and v["tax"]=="roth"}
        if not trad_accts or not roth_accts:
            continue
        roth_name = next(iter(roth_accts))
        total_trad = sum(accounts_bal[k] for k in trad_accts)
        for acct_name in trad_accts:
            fraction = accounts_bal[acct_name]/total_trad
            move = min(conv_amount*fraction, accounts_bal[acct_name])
            accounts_bal[acct_name] -= move
            accounts_bal[roth_name] += move

# -------------------------------
# Gifting (Q4)
# -------------------------------
def handle_gifting(self, accounts, accounts_bal, year_idx):
    gifting_amount = self.gifting
    remaining = gifting_amount
    for acct_type in ["taxable","trust","inherited","traditional","roth"]:
        for acct_name, acct in accounts.items():
            if acct["tax"] != acct_type or remaining <= 0:
                continue
            withdraw_amt = min(accounts_bal[acct_name], remaining)
            accounts_bal[acct_name] -= withdraw_amt
            remaining -= withdraw_amt
            if remaining <= 0:
                break
        if remaining <= 0:
            break

# -------------------------------
# Car replacement / home repair
# -------------------------------
def handle_lumpy_home_car_expenses(self, year_idx, month):
    # Example: car in Q2 month 1
    if month == 4:
        pass  # logic to deduct car cost
    # Home repair random month
    if month == 6:
        pass  # logic to deduct repair

# -------------------------------
# Final income for taxes
# -------------------------------
def compute_final_income(self, accounts, accounts_bal, year_idx):
    final_ordinary = sum(accounts_bal.values())*0.5  # Placeholder
    final_ltcg = sum(accounts_bal.values())*0.1     # Placeholder
    return final_ordinary, final_ltcg

# -------------------------------
# Taxable portion of SS
# -------------------------------
def taxable_ss(self, year_idx):
    return 0.85*24000  # Example

# -------------------------------
# Final taxes
# -------------------------------
def calculate_final_taxes(self, final_ordinary, final_ltcg, year_idx):
    federal_tax = 0.2*(final_ordinary+final_ltcg)
    state_tax = 0.0575*(final_ordinary+final_ltcg)
    medicare = 5000
    return federal_tax, state_tax, medicare
```
