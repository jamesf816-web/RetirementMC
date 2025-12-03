print(">>> USING simulator.py WITH DEBUG PARAM")

import numpy as np
import pandas as pd
from config.user_input import *

from engine.roth_optimizer import optimal_roth_conversion


class RetirementSimulator:
    def __init__(self, n_sims=1000):
        self.n_sims = n_sims
        self.years = np.arange(current_year, current_year + n_years)
        self.ages_JEF = np.arange(current_age_JEF, current_age_JEF + n_years)
        self.ages_SEF = np.arange(current_age_SEF, current_age_SEF + n_years)
        self.portfolio_paths = None
        self.conversion_paths_JEF = None
        self.conversion_paths_SEF = None
        self.account_paths = None

    def generate_returns(self):
        """Generate Monte Carlo equity, bond, and inflation returns."""
        def mr_params(i_mu, lt_mu, i_sig, lt_sig, half_life=10):
            revert = np.log(2) / half_life
            mu_t = lt_mu + (i_mu - lt_mu) * np.exp(-revert * np.arange(n_years))
            sig_t = lt_sig + (i_sig - lt_sig) * np.exp(-revert * np.arange(n_years))
            return mu_t, sig_t

        eq_mu, eq_sig = mr_params(initial_equity_mu, long_term_equity_mu,
                                  initial_equity_sigma, long_term_equity_sigma)
        bo_mu, bo_sig = mr_params(initial_bond_mu, long_term_bond_mu,
                                  initial_bond_sigma, long_term_bond_sigma)
        inf_mu, inf_sig = mr_params(initial_inflation_mu, long_term_inflation_mu,
                                    initial_inflation_sigma, long_term_inflation_sigma)

        L = np.linalg.cholesky(corr_matrix)
        shocks = np.random.randn(self.n_sims, n_years, 3) @ L.T

        equity_r = eq_mu + eq_sig * shocks[:, :, 0]
        bond_r = bo_mu + bo_sig * shocks[:, :, 1]
        inflation = np.maximum(inf_mu + inf_sig * shocks[:, :, 2], -0.01)

        return equity_r, bond_r, inflation

    def run_simulation(self, roth_strategy="fill_24_percent", return_trajectories=False, accounts=None, debug_log=None):
        """
        Run the Monte Carlo retirement simulation.
        """
        if accounts is None:
            raise ValueError("You must pass the 'accounts' dictionary with initial balances and account info.")

        if debug_log is None:
            debug_log = [] 

        equity_r, bond_r, inflation = self.generate_returns()
        n_years_sim = len(self.years)

        # Initialize arrays for trajectories
        portfolio_end = np.zeros(self.n_sims)
        converted_this_year_total = np.zeros(self.n_sims)
        portfolio_paths = np.zeros((self.n_sims, n_years_sim))
        conversion_paths_JEF = np.zeros((self.n_sims, n_years_sim))
        conversion_paths_SEF = np.zeros((self.n_sims, n_years_sim))
        account_names = list(accounts.keys())
        account_paths = {name: np.zeros((self.n_sims, n_years_sim)) for name in account_names}

        # Helper function for Social Security taxable portion
        def taxable_ss(age, start_age, annual_pia):
            return annual_pia * 0.85 if age >= start_age else 0.0

        # --- Main Monte Carlo loop ---
        for sim_idx in range(self.n_sims):
            accounts_bal = {name: acct["balance"] for name, acct in accounts.items()}
            converted_this_year_total[sim_idx] = 0.0

            for y, year in enumerate(self.years):
                # 1. Market growth
                for name, acct in accounts.items():
                    eq = acct["equity"]
                    ret = eq * equity_r[sim_idx, y] + (1 - eq) * bond_r[sim_idx, y]
                    accounts_bal[name] *= (1 + ret)

                # 2. Mandatory distributions / RMDs (trusts, inherited, traditional)
                taxable_income_JEF = 0.0
                taxable_income_SEF = 0.0

                for acct_name, acct in accounts.items():
                    if acct["tax"] in ["traditional", "inherited"] and acct.get("mandatory_yield", 0) > 0:
                        mand = accounts_bal[acct_name] * acct["mandatory_yield"]
                        accounts_bal[acct_name] -= mand
                        if acct["owner"] == "JEF":
                            taxable_income_JEF += mand
                        elif acct["owner"] == "SEF":
                            taxable_income_SEF += mand

                # 3. Other income
                # A. Deferred salary
                for item in deferred_salary:
                    if item["year"] == year:
                        taxable_income_JEF += item.get("amount", 0)

                # B. Social Security
                taxable_income_JEF += taxable_ss(self.ages_JEF[y], social_security["JEF"]["start_age"],
                                                 social_security["JEF"]["annual_pia"])
                taxable_income_SEF += taxable_ss(self.ages_SEF[y], social_security["SEF"]["start_age"],
                                                 social_security["SEF"]["annual_pia"])

                # C. Pension
                if self.ages_JEF[y] >= pension["JEF"]["start_age"]:
                    taxable_income_JEF += pension["JEF"]["annual_amount"]

                # Total MFJ income before Roth
                mfj_income = taxable_income_JEF + taxable_income_SEF

                # 4. Roth conversions
                if roth_strategy != "none":
                    
                    debug_log.append(f"Year {year}: before any Roth: MFJ income = {mfj_income}")

                    person = "SEF"
                    person_accounts = {k: v for k, v in accounts_bal.items()
                    if k.startswith(person) and accounts[k]["tax"] == "traditional"}
                    trad_total = sum(person_accounts.values())
                    if trad_total > 0:
                            conv = optimal_roth_conversion(
                                year=year,
                                traditional_balance=trad_total,
                                taxable_income_before_conv=mfj_income,
                                strategy=roth_strategy
                            )
                            for name, balance in person_accounts.items():
                                fraction = balance / trad_total
                                taken = min(balance, conv * fraction)
                                accounts_bal[name] -= taken
                                roth_name = name.replace("Trad", "Roth")
                                if roth_name not in accounts_bal:
                                    accounts_bal[roth_name] = 0
                                accounts_bal[roth_name] += taken
                                converted_this_year_total[sim_idx] += taken
                                conversion_paths_SEF[sim_idx, y] += taken
                                
                    # DEBUG PRINT: SEF conversions applied, mfj_income updated
                    mfj_income += sum(accounts_bal[k.replace("Trad", "Roth")] for k in person_accounts)
                    debug_log.append(f"Year {year}: after SEF Roth: MFJ income = {mfj_income}")
                                   
                    person = "JEF"
                    person_accounts = {k: v for k, v in accounts_bal.items()
                    if k.startswith(person) and accounts[k]["tax"] == "traditional"}
                    trad_total = sum(person_accounts.values())
                    if trad_total > 0:
                            conv = optimal_roth_conversion(
                                year=year,
                                traditional_balance=trad_total,
                                taxable_income_before_conv=mfj_income,
                                strategy=roth_strategy
                            )
                            for name, balance in person_accounts.items():
                                fraction = balance / trad_total
                                taken = min(balance, conv * fraction)
                                accounts_bal[name] -= taken
                                roth_name = name.replace("Trad", "Roth")
                                if roth_name not in accounts_bal:
                                    accounts_bal[roth_name] = 0
                                accounts_bal[roth_name] += taken
                                converted_this_year_total[sim_idx] += taken
                                conversion_paths_JEF[sim_idx, y] += taken
     
                    # DEBUG PRINT: SEF conversions applied, mfj_income updated
                    mfj_income += sum(accounts_bal[k.replace("Trad", "Roth")] for k in person_accounts)
                    debug_log.append(f"Year {year}: after JEF Roth: MFJ income = {mfj_income}")
                                
                # 5. Spending
                real_spend = base_annual_spending * (1.03 ** y)
                total_port = sum(accounts_bal.values())
                if total_port < real_spend:
                    accounts_bal = {k: 0 for k in accounts_bal}
                    portfolio_paths[sim_idx, y:] = 0
                    conversion_paths_JEF[sim_idx, y:] = 0
                    conversion_paths_SEF[sim_idx, y:] = 0
                    break

                # Withdraw proportionally
                for name in accounts_bal:
                    accounts_bal[name] *= (1 - real_spend / total_port)

                # Store trajectories
                portfolio_paths[sim_idx, y] = sum(accounts_bal.values())
                for name in account_names:
                    account_paths[name][sim_idx, y] = accounts_bal[name]

            # Final portfolio value
            portfolio_end[sim_idx] = sum(accounts_bal.values())

        # 6. Stats
        success_rate = np.mean(portfolio_end > 100_000) * 100

        result = {
            "success_rate": success_rate,
            "median_final": np.median(portfolio_end),
            "p10_final": np.percentile(portfolio_end, 10),
            "total_converted_median": np.median(converted_this_year_total),
            "account_paths": account_paths,
            "conversion_paths_JEF": conversion_paths_JEF,
            "conversion_paths_SEF": conversion_paths_SEF,
            "years": self.years
        }

        if return_trajectories:
            result["portfolio_paths"] = portfolio_paths

        return result
