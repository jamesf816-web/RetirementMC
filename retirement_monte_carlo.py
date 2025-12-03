import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t, lognorm, gamma
import seaborn as sns
import multiprocessing as mp

# =============================================================================
# USER INPUT SECTION - EDIT THIS PART WITH YOUR DATA
# =============================================================================

# Basic info
current_year = 2025
current_age_JEF = 60
current_age_SEF = 66
retirement_age = 60
end_age = 100
n_years = end_age - current_age_JEF + 1
n_simulations = 500

# Inflation & market regime (November 2025 reality)
initial_inflation_mu = 0.035          # starting expected inflation
initial_inflation_sigma = 0.025
long_term_inflation_mu = 0.025
long_term_inflation_sigma = 0.015
years_to_revert = 10                  # mean-reversion speed

initial_equity_mu = 0.06              # lower near-term equity return
initial_equity_sigma = 0.16
long_term_equity_mu = 0.075
long_term_equity_sigma = 0.165

initial_bond_mu = 0.03
initial_bond_sigma = 0.08
long_term_bond_mu = 0.045
long_term_bond_sigma = 0.09

# Correlation matrix (equity, bond, inflation)
corr_matrix = np.array([
    [ 1.00, -0.30,  0.20],   # equity
    [-0.30,  1.00, -0.50],   # bond
    [ 0.20, -0.50,  1.00]    # inflation
])


# Accounts (add as many as you want)
accounts = {
                  "GSTExempt":     {"balance": 3_132_760, "equity": 0.75, "bond": 0.25, "tax": "taxable", "owner": "JEF", "type": "trust", "mandatory_yield": 0.040},
                  "REFIrrev":      {"balance": 1_806_065, "equity": 0.75, "bond": 0.25, "tax": "taxable", "owner": "JEF", "type": "trust", "mandatory_yield": 0.040},
                  "JEFTaxable":    {"balance": 1_347_857, "equity": 0.92, "bond": 0.08, "tax": "taxable", "owner": "JEF"},
                  "JointTaxable":  {"balance":   857_710, "equity": 0.26, "bond": 0.74, "tax": "taxable", "owner": "JEF"},
                  "JEF_RoloverIRA":{"balance":   854_793, "equity": 0.77, "bond": 0.23, "tax": "traditional", "owner": "JEF"},
                  "JEF_TradIRA":   {"balance":   255_797, "equity": 0.82, "bond": 0.18, "tax": "traditional", "owner": "JEF"},
                  "InheritIRA63":  {"balance":   915_953, "equity": 0.81, "bond": 0.19, "tax": "inherited", "owner": "JEF", "rmd_factor_table": "IRS_Single_Life"},
                  "InheritIRA77":  {"balance":   175_130, "equity": 0.78, "bond": 0.22, "tax": "inherited", "owner": "JEF", "rmd_factor_table": "IRS_Single_Life"},
                  "JEF_TIAA":      {"balance":   367_118, "equity": 0.81, "bond": 0.19, "tax": "traditional", "owner": "JEF"},
                  "SEF_TradIRA":   {"balance":   212_170, "equity": 1.00, "bond": 0.00, "tax": "traditional", "owner": "SEF"},
                  "JEF_Roth":      {"balance":   172_451, "equity": 1.00, "bond": 0.00, "tax": "roth", "owner": "JEF"},
                  "SEF_Roth":      {"balance":    60_726, "equity": 0.85, "bond": 0.15, "tax": "roth", "owner": "SEF"},
                          }

# Income streams
deferred_salary = [
                  {"year": 2026, "amount": 6_250},  # one-time or few years
                  {"year": 2027, "amount": 6_250},
                  {"year": 2028, "amount": 6_250},
                  {"year": 2029, "amount": 6_250},
                  {"year": 2030, "amount": 6_250},
                      ]

social_security = {
                  "JEF": {"start_age": 66, "annual_pia": 9_000},  # at FRA
                  "SEF": {"start_age": 70, "annual_pia": 70_000},
                          "cola": True
                          }

pension = {
                  "JEF":{"start_age": 65, "annual_amount": 31_872},
                          "cola": False
                          }

# Expenses
base_annual_spending = 175_800         # today's dollar
medicare_start_age = 65
medicare_supplement_annual = 5_000     # extra cost starting at Medicare age

mortgage_payoff_year = 2031
mortgage_monthly_until_payoff = 3_625 * 12

# Intermittent expenses
car_replacement_cycle = 8               # every 10 years
car_cost_today = 50_000
car_inflation = 0.04                    # cars inflate faster

home_repair_mean = 25_000               # average major repair
home_repair_shape = 2.0                 # gamma shape (for stochastic)

# Withdrawal & tax strategy
initial_withdrawal_rate = 0.05          # we will optimize this
withdrawal_floor = 175_000              # optional dollar floor (TBD)
withdrawal_ceiling = 600_000

roth_conversion_target_bracket_edges = [102_135, 191_950, 364_200, 462_500]
max_conversion_to_stay_under_irmaa2 = 250_000

standard_deduction_mfj = 30_000
long_term_cap_gains_brackets_mfj = [0, 94_050, 583_750]

def get_rmd_factor(age, year):
    """
    Returns the IRS Uniform Lifetime Table divisor for a given age and year.
    After the SECURE 2.0 Act, the table was updated starting 2022 and changes slightly over time.
    This function uses the post-2022 table (applicable through at least 2032).
    For spouse >10 years younger we could add the Joint table, but MFJ same age is very close.
    """
    # IRS Uniform Lifetime Table (2022 onward) — age = divisor
    uniform_table = {
        73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1, 80: 20.2,
        81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2, 87: 14.4, 88: 13.7,
        89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5,  95: 8.9,  96: 8.4,
        97: 7.8,  98: 7.3,  99: 6.8, 100: 6.4, 101: 6.0, 102: 5.6, 103: 5.2, 104: 4.9,
        105: 4.6, 106: 4.3, 107: 4.1, 108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4, 112: 3.3,
        113: 3.1, 114: 3.0, 115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3, 120: 2.0
    }
    # For ages 120+ IRS says use 2.0, but we'll cap it
    return uniform_table.get(age, 2.0)

# IRS Single Life Expectancy Table (2025+ updated tables – divisor decreases by ~1 each year)
IRS_SINGLE_LIFE = {
    35:53.3, 36:52.3, 37:51.3, 38:50.3, 39:49.3, 40:48.3, 41:47.3, 42:46.3, 43:45.3, 44:44.3,
    45:43.3, 46:42.4, 47:41.4, 48:40.4, 49:39.4, 50:38.4, 51:37.4, 52:36.4, 53:35.4, 54:34.4,
    55:33.4, 56:32.4, 57:31.4, 58:30.4, 59:29.4, 60:28.4, 61:27.4, 62:26.4, 63:25.4, 64:24.4,
    65:23.5, 66:22.5, 67:21.5, 68:20.5, 69:19.5, 70:18.5, 71:17.6, 72:16.6, 73:15.6, 74:14.7,
    75:13.8, 76:12.9, 77:12.0, 78:11.2, 79:10.4, 80:9.6,  81:8.9,  82:8.2,  83:7.5,  84:6.9,
    85:6.3,  86:5.8,  87:5.3,  88:4.9,  89:4.5,  90:4.1,  91:3.8,  92:3.5,  93:3.2,  94:3.0,
    95:2.8,  96:2.6,  97:2.4,  98:2.3,  99:2.1, 100:2.0, 101:1.9, 102:1.8, 103:1.7, 104:1.6,
    105:1.5, 106:1.4, 107:1.3, 108:1.2, 109:1.1, 110:1.0, 111:0.9, 112:0.8, 113:0.7, 114:0.6,
    115:0.5, 116:0.4, 117:0.3, 118:0.2, 119:0.1, 120:0.1
}


# =============================================================================
# TAX & IRMAA INFLATION PARAMETERS (2025 baseline)
# =============================================================================
tax_bracket_inflation_rate = 0.025     # historical average ~2.5%
standard_deduction_mfj_2025 = 30_000   # projected 2025 (actual will be ~29,900–30,200)
standard_deduction_single_2025 = 15_000

# 2025 IRMAA brackets (Married Filing Jointly) — includes the new 6th tier starting 2026 but we’ll use 2025 for 2025
irmaa_brackets_mfj_2025 = [0, 206_000, 258_000, 322_000, 386_000, 750_000]   # upper bounds (actually thresholds are lower bound of next)
irmaa_premium_adders_part_b_2025 = [0, 69.90*12, 174.70*12, 279.50*12, 384.30*12, 419.00*12]   # annual extra
irmaa_premium_adders_part_d_2025 = [0, 12.90*12, 33.30*12, 53.80*12, 74.20*12, 81.00*12]

# We'll inflate IRMAA brackets at ~2.8% per year (slightly higher than CPI because Medicare uses special rule)
irmaa_inflation_rate = 0.028

def get_tax_and_irmaa(year, magi, filing_status="mfj"):
    """
    Returns (federal_income_tax, part_b_irmaa_annual, part_d_irmaa_annual)
    for a given year and MAGI.
    """
    # === Inflate tax brackets and standard deduction from 2025 base ===
    years_since_2025 = year - 2025
    inflation_factor = (1 + tax_bracket_inflation_rate) ** years_since_2025

    std_ded = standard_deduction_mfj_2025 * inflation_factor if filing_status == "mfj" else standard_deduction_single_2025 * inflation_factor

    # 2025 10–37% brackets (MFJ)
    base_brackets = [0, 23_050, 93_850, 201_050, 383_900, 487_450, 731_200]
    rates = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]

    inflated_brackets = [b * inflation_factor for b in base_brackets]

    taxable = max(magi - std_ded, 0)
    tax = 0.0
    prev = 0.0
    for b, r in zip(inflated_brackets[1:], rates[1:]):
        tax += r * max(min(taxable, b) - prev, 0)
        prev = b
    tax += rates[0] * min(taxable, inflated_brackets[1])  # 10% on first slice

    # === IRMAA — uses MAGI from TWO years ago ===
    irmaa_year = year - 2
    if irmaa_year < 2025:
        irmaa_year = 2025  # cap at 2025 rules for early years

    irmaa_infl_factor = (1 + irmaa_inflation_rate) ** (irmaa_year - 2025)
    irmaa_thresholds = [x * irmaa_infl_factor for x in irmaa_brackets_mfj_2025]

    tier = next((i for i, thresh in enumerate(irmaa_thresholds) if magi <= thresh), len(irmaa_thresholds)-1)

    part_b_adder = irmaa_premium_adders_part_b_2025[tier] * irmaa_infl_factor
    part_d_adder = irmaa_premium_adders_part_d_2025[tier] * irmaa_infl_factor

    return tax, part_b_adder, part_d_adder

# =============================================================================
# CORE SIMULATION ENGINE
# =============================================================================

def generate_correlated_returns(n_years, corr_matrix, equity_params, bond_params, infl_params):
    means = np.array([equity_params['mu'], bond_params['mu'], infl_params['mu']])
    cov = corr_matrix * np.outer(np.array([equity_params['sigma'], bond_params['sigma'], infl_params['sigma']]),
                                 np.array([equity_params['sigma'], bond_params['sigma'], infl_params['sigma']]))
    returns = np.random.multivariate_normal(means, cov, size=n_years)
    eq, bd, infl = returns[:,0], returns[:,1], returns[:,2]
    return eq, bd, infl

def run_one_simulation():
    portfolio = {name: acct["balance"] for name, acct in accounts.items()}
    
    results = pd.DataFrame(index=range(current_year, current_year + n_years),
                           columns=["total_assets", "spending", "portfolio_withdrawal", "gross_withdrawl_strategy",
                                    "roth_conversion",
                                    "taxable_income", "tax", "inflation", "equity_return", "bond_return",
                                    "cash_balance", "rmd",
                                    "irmaa_total", "magi_this_year", "magi_for_irmaa"], dtype=float)

    results["roth_conversion"] = 0
    results["withdrawal"] = 0
    results["withdrawal_rate"] = 0
    results["guardrail_floor_hit"] = False
    results["guardrail_ceiling_hit"] = False
    results["federal_tax_bracket"] = 0.0
    results["marginal_rate"] = 0.0
    results["irmaa_bracket"] = 0
    results["ss_taxable_pct"] = 0.0
    results["agi"] = 0.0
    results["taxable_income"] = 0.0

    magi_history = {}
    # Pre-load the two look-back years so IRMAA is correct in the first two retirement years
    magi_history[current_year - 2] = 314_000   # change to your real 2023 MAGI
    magi_history[current_year - 1] = 366_000   # change to your real 2024 MAGI

    
    cash_buffer = 0

    for i, year in enumerate(results.index):
        age_JEF = current_age_JEF + i
        age_SEF = current_age_SEF + i


# ==================================================================
# REQUIRED MINIMUM DISTRIBUTIONS (2025 rules – starts at 73 or 75)
# ==================================================================
        rmd_this_year = 0.0
        ordinary_income = 0.0
        total_converted_this_year = 0.0

        for owner, age in [("JEF", age_JEF), ("SEF", age_SEF)]:
            # Determine if RMD age has been reached this year
            rmd_start_age = 75 if (current_year + i - (current_year - current_age_JEF if owner == "JEF" else current_age_SEF)) >= 1959 else 73
            # Translation: born 1959 or later → 75, born 1951–1958 → 73
            if age < rmd_start_age:
                continue

            # Sum all Traditional (pre-tax) balances for this owner at start of year
            owner_trad_balance = 0.0
            owner_trad_accounts = []

            for name, data in accounts.items():
                if data.get("owner") == owner and data.get("tax") == "traditional" and portfolio.get(name, 0) > 0:
                    owner_trad_balance += portfolio[name]
                    owner_trad_accounts.append(name)

            if owner_trad_balance <= 0:
                continue

            divisor = get_rmd_factor(age, year)
            required = owner_trad_balance / divisor

            # Take the RMD proportionally from this owner's Traditional accounts
            for name in owner_trad_accounts:
                if required <= 0:
                    break
                take = min(portfolio[name], required * (portfolio[name] / owner_trad_balance))
                portfolio[name] -= take
                rmd_this_year += take
                required -= take

        # RMDs are fully taxable ordinary income
        ordinary_income += rmd_this_year

        # Record it (optional – nice for debugging)
        results.loc[year, "rmd"] = rmd_this_year

        
        # Market regime mean reversion
        t = min(i / years_to_revert, 1.0)
        equity_mu  = initial_equity_mu  * (1-t) + long_term_equity_mu  * t
        equity_sig = initial_equity_sigma * (1-t) + long_term_equity_sigma * t
        bond_mu    = initial_bond_mu    * (1-t) + long_term_bond_mu    * t
        bond_sig   = initial_bond_sigma * (1-t) + long_term_bond_sigma * t
        infl_mu    = initial_inflation_mu * (1-t) + long_term_inflation_mu * t
        infl_sig   = initial_inflation_sigma * (1-t) + long_term_inflation_sigma * t

# ---- Generate correlated returns for this single year ----
        eq_ret, bd_ret, inflation = generate_correlated_returns(
            n_years=1,
            corr_matrix=corr_matrix,
            equity_params={'mu': equity_mu,  'sigma': equity_sig},
            bond_params  ={'mu': bond_mu,    'sigma': bond_sig},
            infl_params  ={'mu': infl_mu,    'sigma': infl_sig}
        )

        # Since we asked for 1 year, each is an array of length 1 → extract the scalar
        eq_ret   = eq_ret[0]
        bd_ret   = bd_ret[0]
        inflation = inflation[0]

        # Mild guard against deflation going too negative (optional but nice)
        inflation = max(inflation, -0.02)

        results.loc[year, "inflation"] = inflation
        results.loc[year, "equity_return"] = eq_ret
        results.loc[year, "bond_return"] = bd_ret

        # Grow portfolios
        for name, acct in accounts.items():
            alloc_eq = acct["equity"]
            portfolio[name] *= (1 + alloc_eq * eq_ret + (1-alloc_eq) * bd_ret)

        total_assets = sum(portfolio.values()) + cash_buffer
        results.loc[year, "total_assets"] = total_assets

        # Income sources
        ordinary_income = 0
        deferred = sum(d["amount"] for d in deferred_salary if d["year"] == year)
        ordinary_income += deferred

        ss_income = 0
        if age_JEF >= social_security["JEF"]["start_age"]:
            ss_income += social_security["JEF"]["annual_pia"]
        if age_SEF >= social_security["SEF"]["start_age"]:
            ss_income += social_security["SEF"]["annual_pia"]
        if social_security["cola"] and i > 0:
            ss_income *= (1 + results.loc[year-1, "inflation"])

        pension_income = 0
        if age_JEF >= pension["JEF"]["start_age"]:
            pension_income += pension["JEF"]["annual_amount"]
        if pension["cola"] and i > 0:
            pension_income *= (1 + results.loc[year-1, "inflation"])

        ordinary_income += ss_income + pension_income

        # Expenses in nominal dollars
        real_spending = base_annual_spending * (1 + long_term_inflation_mu) ** i
        cum_infl = np.prod(1 + results.loc[results.index < year, "inflation"]) if i > 0 else 1.0
        nominal_spending = real_spending * cum_infl

        if year < mortgage_payoff_year:
            nominal_spending += mortgage_monthly_until_payoff * (1 + long_term_inflation_mu) ** i

        if current_age_JEF >= medicare_start_age:
            nominal_spending += medicare_supplement_annual * (1 + long_term_inflation_mu) ** i

        if (i) % car_replacement_cycle == 3:
            car_cost = car_cost_today * (1 + car_inflation) ** i
            nominal_spending += car_cost

        if np.random.rand() < 0.20:
            repair = gamma.rvs(home_repair_shape, scale=home_repair_mean/home_repair_shape)
            nominal_spending += repair

# ==================================================================
# GUARDRAILS WITHDRAWAL STRATEGY (5% of prior year with floor/ceiling)
# ==================================================================
        if i == 0:
            # First year: just spend whatever you need (no prior balance yet)
            withdrawal_needed = nominal_spending
        else:
            prior_assets = results.loc[year-1, "total_assets"]

            # Base rule: 5% of prior year-end balance
            candidate = prior_assets * initial_withdrawal_rate

            # Inflate floor and ceiling with cumulative inflation up to last year
            cum_infl_to_last_year = np.prod(1 + results.loc[results.index < year, "inflation"]) if i > 1 else 1.0

            floor_this_year = withdrawal_floor * cum_infl_to_last_year
            ceiling_this_year = withdrawal_ceiling * cum_infl_to_last_year

            # Apply guardrails
            guarded = max(floor_this_year, min(candidate, ceiling_this_year))

            # Mandatory income that must be spent anyway
            mandatory_income = ss_income + pension_income + rmd_this_year   # already calculated above

            # If mandatory income alone already exceeds the ceiling → withdraw nothing from portfolio
            if mandatory_income >= ceiling_this_year:
                withdrawal_needed = 0.0
            else:
                # You still need the full guarded amount, but we can reduce portfolio withdrawal
                # by whatever mandatory income is already covering
                withdrawal_needed = max(0, guarded - mandatory_income)

        # Record the gross amount the strategy wanted (before any mandatory offset)
        results.loc[year, "gross_withdrawal_strategy"] = withdrawal_needed + max(0, mandatory_income - withdrawal_needed if i > 0 else 0)
        results.loc[year, "portfolio_withdrawal"]     = withdrawal_needed   # what actually comes out of investments

# ==================================================================
# PORTFOLIO WITHDRAWAL — define early so tax/IRMAA can use it
        if i == 0:
            portfolio_withdrawal = nominal_spending          # Year 1: whatever you need
        else:
            portfolio_withdrawal = withdrawal_needed         # From guardrails logic

        # <<< ADD THESE TWO LINES — crucial! >>>
        total_converted_this_year = total_converted_this_year if 'total_converted_this_year' in locals() else 0.0
        rmd_this_year = rmd_this_year if 'rmd_this_year' in locals() else 0.0   # in case RMD block hasn't run yet

        # ==================================================================
        # CALCULATE MAGI (for next-next-year's IRMAA)
        # ==================================================================
        magi_this_year = ordinary_income + total_converted_this_year + portfolio_withdrawal * 0.5
        magi_history[year] = magi_this_year

        # ==================================================================
        # FEDERAL TAX + IRMAA (using MAGI from two years ago)
        # ==================================================================
        irmaa_magi = magi_history.get(year - 2, magi_this_year)

        federal_tax, irmaa_b, irmaa_d = get_tax_and_irmaa(year, irmaa_magi, filing_status="mfj")

        nominal_spending += irmaa_b + irmaa_d   # IRMAA hits spending this year

        # Record everything
        results.loc[year, "tax"]                = federal_tax
        results.loc[year, "irmaa_total"] = irmaa_b + irmaa_d
        results.loc[year, "magi_for_irmaa"] = irmaa_magi
        results.loc[year, "magi_this_year"] = magi_this_year
        results.loc[year, "portfolio_withdrawal"] = portfolio_withdrawal
        
# ==================================================================
# ==== Smart Roth conversion – respects ownership and multiple accounts ====
# ==================================================================
        conversion_target_income = min(191_950, max_conversion_to_stay_under_irmaa2)
        remaining_room = max(conversion_target_income - ordinary_income, 0)

        total_converted_this_year = 0

        if remaining_room > 5000:                                      # only if real money
            # Priority: wife first (older), then you
            for owner in ["SEF", "JEF"]:
                if remaining_room <= 5000:
                    break

                # Collect all Traditional accounts belonging to this owner
                trad_accounts = [name for name, data in accounts.items()
                                 if portfolio.get(name, 0) > 0 and data["tax"] == "traditional" and data["owner"] == owner]

                for acct_name in trad_accounts:
                    if remaining_room <= 5000:
                        break

                    can_convert = min(portfolio[acct_name], remaining_room)

                    # Perform the conversion
                    portfolio[acct_name] -= can_convert

                    # Add to this owner’s Roth
                    if owner == "SEF":
                        roth_name = "SEF_Roth"
                    else:
                        roth_name = "JEF_Roth"

                    portfolio[roth_name] = portfolio.get(roth_name, 0) + can_convert

                    ordinary_income += can_convert
                    remaining_room -= can_convert
                    total_converted_this_year += can_convert

                    results.loc[year, "roth_conversion"] = total_converted_this_year


# ==================================================================
# CASH SHORTFALL – smart withdrawal order (2025 real life version)
# ==================================================================
        shortfall = 0
        total_withdrawal_this_year = 0

        if cash_buffer < 30_000:
            shortfall = -cash_buffer

        if shortfall > 0 or any(acct.get("type") == "trust" for acct in accounts.values()):
            # 1. First: Forced minimum distributions from the trusts (interest + dividends)
            mandatory_trust_pull = 0
            for name, data in accounts.items():
                if data.get("type") != "trust" or portfolio[name] <= 0:
                    continue
                mand = data["mandatory_yield"] * data["balance"]   # original balance
                take = min(mand * 1.2, portfolio[name])           # allow 20% buffer for growth
                portfolio[name] -= take
                mandatory_trust_pull += take

            cash_buffer += mandatory_trust_pull
            shortfall = -cash_buffer   # recalculate after mandatory pull

            # 2. If still short (or we just want clean numbers), pull the rest proportionally
            #     from: regular brokerage + trusts (extra principal)
            if shortfall > 0:
                total_withdrawal_this_year = shortfall

                # Define buckets we are willing to pull extra from (brokerage + trusts)
                flexible_taxable_buckets = [n for n, d in accounts.items()
                                            if d["tax"] == "taxable" and portfolio.get(n, 0) > 100]

                # How much of the extra should come from trusts vs regular brokerage?
                # Option A: 50/50 split (change the ratio to whatever you prefer)
                trust_ratio = 0.50

                trust_flex   = [n for n in flexible_taxable_buckets if accounts[n].get("type") == "trust"]
                broker_flex  = [n for n in flexible_taxable_buckets if accounts[n].get("type") != "trust"]

                # Pull from trusts first (extra principal)
                trust_extra = 0
                for name in trust_flex:
                    if shortfall <= 0: break
                    proportion = portfolio[name] / sum(portfolio.get(n,0) for n in trust_flex) if trust_flex else 0
                    take = min(portfolio[name], shortfall * trust_ratio * proportion + 1000)
                    portfolio[name] -= take
                    trust_extra += take
                    shortfall -= take

                # Pull the rest from regular brokerage
                for name in broker_flex:
                    if shortfall <= 0: break
                    proportion = portfolio[name] / sum(portfolio.get(n,0) for n in broker_flex) if broker_flex else 0
                    take = min(portfolio[name], shortfall * proportion + 1000)
                    portfolio[name] -= take
                    shortfall -= take

                # Absolute last resort – Traditional IRAs (very tax-inefficient)
                if shortfall > 0:
                    trads = [n for n, d in accounts.items() if d["tax"] == "traditional"]
                    for name in trads:
                        if shortfall <= 0: break
                        take = min(portfolio[name], shortfall)
                        portfolio[name] -= take
                        ordinary_income += take
                        shortfall -= take

            cash_buffer += total_withdrawal_this_year

        results.loc[year, "portfolio_withdrawal"] = withdrawal_needed
        results.loc[year, "total_withdrawal"]     = withdrawal_needed + total_withdrawal_this_year + mandatory_trust_pull
        results.loc[year, "trust_mandatory"] = mandatory_trust_pull
        results.loc[year, "spending"] = nominal_spending
        results.loc[year, "roth_conversion"] = total_converted_this_year
        results.loc[year, "cash_balance"] = cash_buffer

        if total_assets <= 0:
            results.loc[year:, "total_assets"] = 0
            break

        if total_assets <= 0:
            # <<< THIS IS THE ONLY LINE YOU NEED TO ADD / FIX >>>
            results.loc[year:, :] = 0                      # ← zeros EVERY column for remaining years
            break
        
    return results

# =============================================================================
# RUN SIMULATIONS (parallel)
# =============================================================================

def _run_sim_wrapper(_):
    return run_one_simulation()

if __name__ == "__main__":
    with mp.Pool(mp.cpu_count() - 1) as pool:  # leave 1 core free
        sims = pool.map(_run_sim_wrapper, range(n_simulations))
        
        all_assets = pd.concat([sim["total_assets"].rename(f"sim.{i}") for i, sim in enumerate(sims)], axis=1)

        success_rate = (all_assets.iloc[-1] > 0).mean()
        p10 = all_assets.quantile(0.10, axis=1)
        p50 = all_assets.quantile(0.50, axis=1)
        p90 = all_assets.quantile(0.90, axis=1)

        print(f"\nSuccess rate: {success_rate:.1%}")
        print(f"Median ending balance: ${p50.iloc[-1]:,.0f}")
        print(f"10th percentile ending balance: ${p10.iloc[-1]:,.0f}")

        plt.figure(figsize=(12,7))
        plt.plot(all_assets.index, p10/1e6, label="10th percentile", color="red")
        plt.plot(all_assets.index, p50/1e6, label="Median", linewidth=2)
        plt.plot(all_assets.index, p90/1e6, label="90th percentile", color="green")
        plt.fill_between(all_assets.index, p10/1e6, p90/1e6, alpha=0.2)
        plt.title(f"Monte Carlo Retirement – {n_simulations:,} paths")
        plt.ylabel("Portfolio (millions $)")
        plt.xlabel("Year")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

        all_results.xs("roth_conversion", axis=1, level=1).plot()
        withdrawals = all_results.xs("withdrawal_rate", axis=1, level=1)
        withdrawals.quantile([0.1, 0.5, 0.9], axis=1).T.plot()
