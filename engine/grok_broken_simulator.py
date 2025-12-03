import numpy as np
import pandas as pd
import copy
import math

from config.user_input import (
    current_year, current_age_person1, current_age_person2, retirement_age, end_age, n_years, n_simulations
)
from config.market_assumptions import *
from config.expense_assumptions import *
from config.default_portfolio import *
from engine.rmd_tables import get_rmd_factor
from engine.tax_utils import get_current_brackets
from engine.roth_optimizer import optimal_roth_conversion
from engine.tax_engine import calculate_taxes


class RetirementSimulator:
    def __init__(self, n_sims=1000):
        self.n_sims = n_sims
        self.n_years = n_years
        self.years = np.arange(current_year - 2, current_year + n_years)
        self.ages_person1 = np.arange(current_age_person1 - 2, current_age_person1 + n_years)
        self.ages_person2 = np.arange(current_age_person2 - 2, current_age_person2 + n_years)

        # === Properly read from current config/ structure ===
        self.pension_age       = pension["person1"]["start_age"]
        self.pension_amount    = pension["person1"]["annual_amount"]
        self.pension_cola      = pension["cola"]
        self.ss_age_person1        = social_security["person1"]["start_age"]
        self.ss_age_person2        = social_security["person2"]["start_age"]
        self.ss_amount_person1     = social_security["person1"]["annual_pia"]
        self.ss_amount_person2     = social_security["person2"]["annual_pia"]
        self.defered_salary_map = {item["year"]: item["amount"] for item in deferred_salary}

        # Market parameters
        self.initial_equity_mu = initial_equity_mu
        self.long_term_equity_mu = long_term_equity_mu
        self.initial_equity_sigma = initial_equity_sigma
        self.long_term_equity_sigma = long_term_equity_sigma
        self.initial_bond_mu = initial_bond_mu
        self.long_term_bond_mu = long_term_bond_mu
        self.initial_bond_sigma = initial_bond_sigma
        self.long_term_bond_sigma = long_term_bond_sigma
        self.initial_inflation_mu = initial_inflation_mu
        self.long_term_inflation_mu = long_term_inflation_mu
        self.initial_inflation_sigma = initial_inflation_sigma
        self.long_term_inflation_sigma = long_term_inflation_sigma
        self.corr_matrix = corr_matrix

        # Home repair parameters
        self.home_repair_prob = home_repair_prob
        self.home_repair_mean = home_repair_mean
        self.home_repair_shape = home_repair_shape

        # Pre-allocate path arrays (will be initialized in run_simulation)
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
        self.pension_paths = None
        self.medicare_paths = None
        self.defered_salary_map = {item["year"]: item["amount"] for item in deferred_salary}

        # Store imported parameters as instance variables
        self.initial_equity_mu = initial_equity_mu
        self.long_term_equity_mu = long_term_equity_mu
        self.initial_equity_sigma = initial_equity_sigma
        self.long_term_equity_sigma = long_term_equity_sigma

        self.initial_bond_mu = initial_bond_mu
        self.long_term_bond_mu = long_term_bond_mu
        self.initial_bond_sigma = initial_bond_sigma
        self.long_term_bond_sigma = long_term_bond_sigma

        self.initial_inflation_mu = initial_inflation_mu
        self.long_term_inflation_mu = long_term_inflation_mu
        self.initial_inflation_sigma = initial_inflation_sigma
        self.long_term_inflation_sigma = long_term_inflation_sigma

        self.corr_matrix = corr_matrix

    def generate_returns(self, n_full):
        """Generate Monte Carlo equity, bond, and inflation returns."""
        def mr_params(i_mu, lt_mu, i_sig, lt_sig, half_life=10):
            revert = np.log(2) / half_life
            mu_t = lt_mu + (i_mu - lt_mu) * np.exp(-revert * np.arange(n_full))
            sig_t = lt_sig + (i_sig - lt_sig) * np.exp(-revert * np.arange(n_full))
            return mu_t, sig_t

        eq_mu, eq_sig = mr_params(self.initial_equity_mu, self.long_term_equity_mu,
                                  self.initial_equity_sigma, self.long_term_equity_sigma)
        bo_mu, bo_sig = mr_params(self.initial_bond_mu, self.long_term_bond_mu,
                                  self.initial_bond_sigma, self.long_term_bond_sigma)
        inf_mu, inf_sig = mr_params(self.initial_inflation_mu, self.long_term_inflation_mu,
                                    self.initial_inflation_sigma, self.long_term_inflation_sigma)

        L = np.linalg.cholesky(self.corr_matrix)
        shocks = np.random.randn(self.n_sims, n_full, 3) @ L.T

        # calculate quarterly returns and inflation
        eq_mu_q = (1 + eq_mu) ** 0.25 -1
        eq_sig_q = eq_sig / np.sqrt(4)
        bo_mu_q = (1 + bo_mu) ** 0.25 -1
        bo_sig_q = bo_sig / np.sqrt(4)
        inf_mu_q = (1 + inf_mu) ** 0.25 -1
        inf_sig_q = inf_sig / np.sqrt(4)

        n_quarters = n_full * 4
        eq_mu_q = np.repeat(eq_mu_q, 4)
        bo_mu_q = np.repeat(bo_mu_q, 4)
        inf_mu_q = np.repeat(inf_mu_q, 4)
        eq_sig_q = np.repeat(eq_sig_q, 4)
        bo_sig_q = np.repeat(bo_sig_q, 4)
        inf_sig_q = np.repeat(inf_sig_q, 4)
        # Shocks done quarterly rather than annually
        L = np.linalg.cholesky(self.corr_matrix)
        shocks = np.random.randn(self.n_sims, n_quarters, 3) @ L.T

        equity_r_q = eq_mu_q + eq_sig_q * shocks[:, :, 0]
        bond_r_q = bo_mu_q + bo_sig_q * shocks[:, :, 1]
        infl_q = np.maximum(inf_mu_q + inf_sig_q * shocks[:, :, 2], -0.01/4) # cap deflation
        
        # return equity_r, bond_r, inflation # using quarterly returns now
        return equity_r_q, bond_r_q, infl_q


    def sample_home_repair_cost(self, rng: np.random.Generator) -> float:
        """
        Returns the home repair cost for ONE year.
        Most years: $0
        Some years: a large skewed cost (lognormal)
        """
        # Pre-compute lognormal parameters for major home repairs
        mu_mhr = np.log(home_repair_mean / np.sqrt(1 + home_repair_shape**2))
        sigma_mhr = np.sqrt(np.log(1 + home_repair_shape**2))

        if rng.random() < home_repair_prob:
            # Lognormal draw: mean = home_repair_mean, cv ≈ home_repair_shape
            cost_mhr = rng.lognormal(mean=mu_mhr, sigma=sigma_mhr)
            return round(cost_mhr, 2)
        else:
            return 0.0

    def get_roth_accounts(self, accounts, person):
        return {
            name: acct
            for name, acct in accounts.items()
            if acct.get("tax") == "roth" and acct.get("owner") == person
        }

    def estimate_taxable_gap(self, cash_needed, accounts, accounts_bal, inflation_this_year):
        """
        Estimates the taxable portion of cash needed to cover essentials/travel.
        Returns (ordinary_income, ltcg_income)
        """
        ordinary_income = 0.0
        ltcg_income = 0.0

        spending_need = cash_needed
        essential_withdrawn_total = 0.0
        remaining = spending_need - essential_withdrawn_total
        withdrawal_order = ["taxable", "trust", "inherited", "traditional", "roth"]

        est_tax_bal = accounts_bal.copy()   # copy so doesn't modify real balances

        for acct_type in withdrawal_order:
           for acct_name, acct in accounts.items():
               if acct["tax"] != acct_type:
                   continue
               if remaining <= 0:
                   break

               # Determine how much to withdraw
               acct_balance = est_tax_bal[acct_name]
               withdraw_amt = min(acct_balance, remaining)

               # Track taxable portions
               if acct_type == "taxable":
                   # calculate percentage that is unrealized gains
                   current_gain = acct["balance"] - acct["basis"]
                   if current_gain >0:
                       gain_percentage = current_gain / acct["balance"]
                       realized_gains = withdraw_amt * gain_percentage
                   else:
                       realized_gains = 0.0
                   # Split TAXABLE portion[realized_gains] between ordinary vs LTCG
                   ordinary_part = realized_gains * acct.get("ordinary_pct", 0.1) #this assumes a portfolio heavily weighed to stocks (LTCG)
                   ltcg_part = realized_gains - ordinary_part
                   ordinary_income += ordinary_part
                   ltcg_income += ltcg_part
               elif acct_type == "trust":
                   ordinary_income += 0 # Trust principle draws are not taxable income                           
               elif acct_type == "roth":
                   ordinary_income += 0 # ROTH draws are not taxable income                           
               else: # taxable IRA draws
                   ordinary_income += withdraw_amt

               est_tax_bal[acct_name] -= withdraw_amt
               essential_withdrawn_total += withdraw_amt
               remaining -= withdraw_amt

        return ordinary_income, ltcg_income

    def run_simulation(self, tax_strategy="fill_24_percent", irmaa_strategy="IRMAA_Tier4", max_roth=250000, base_annual_spending=130000, withdrawal_rate=0.03, travel=50000, gifting=50000, return_trajectories=False, accounts=None, debug_log=None):
        """
        Run the Monte Carlo retirement simulation.
        """
        if accounts is None:
            raise ValueError("You must pass the 'accounts' dictionary with initial balances and account info.")

        if debug_log is None:
            debug_log = [] 

        # Extracted: Initialize all arrays
        self._initialize_arrays(accounts)

        # Extracted: Generate market returns (quarterly)
        equity_r_q, bond_r_q, infl_q = self._generate_market_returns(self.n_years)

        # Loop over simulations
        for sim_idx in range(self.n_sims):
            self._run_single_simulation(
                sim_idx, equity_r_q, bond_r_q, infl_q,
                tax_strategy, irmaa_strategy, max_roth,
                base_annual_spending, withdrawal_rate, travel, gifting,
                accounts, debug_log
            )

        # 7. Stats (unchanged)
        threshold = 500_000
        portfolio_end = self.portfolio_paths[:, -1]
        success_rate = np.mean(portfolio_end > threshold) * 100
        minimum_annual_balance = np.min(self.portfolio_paths, axis=1)
        avoid_ruin_rate = np.mean(minimum_annual_balance > threshold) *100
        

        result = {
            "success_rate": success_rate,
            "avoid_ruin_rate": avoid_ruin_rate,
            "median_final": np.median(portfolio_end),
            "p10_final": np.percentile(portfolio_end, 10),
            "account_paths": self.account_paths,
            "conversion_paths": self.conversion_paths,
            "travel_paths": self.travel_paths,  # Assuming travel_paths is defined in config or elsewhere; if not, adjust
            "gifting_paths": self.gifting_paths,  # Same as above
            "base_spending_paths": self.base_spending_paths,
            "lumpy_spending_paths": self.lumpy_spending_paths,
            "plan_paths": self.plan_paths,
            "taxes_paths": self.taxes_paths,
            "magi_paths": self.magi_paths,
            "trust_income_paths": self.trust_income_paths,
            "ssbenefit_paths": self.ssbenefit_paths,
            "portfolio_withdrawal_paths": self.portfolio_withdrawal_paths,
            "rmd_paths": self.rmd_paths,
            "pension_paths": self.pension_paths,
            "medicare_paths": self.medicare_paths,
            "years": self.years
        }

        if return_trajectories:
            result["portfolio_paths"] = self.portfolio_paths

        return result

    # --- Extracted Methods ---

    def _initialize_arrays(self, accounts):
        """Initialize all simulation arrays."""
        n_full = self.n_years + 2  # Account for pre-2026 years
        self.portfolio_paths = np.zeros((self.n_sims, n_full))
        self.conversion_paths = np.zeros((self.n_sims, n_full))
        self.account_paths = {name: np.zeros((self.n_sims, n_full)) for name in accounts.keys()}
        self.plan_paths = np.zeros((self.n_sims, n_full))
        self.travel_paths = np.zeros((self.n_sims, n_full))
        self.gifting_paths = np.zeros((self.n_sims, n_full))
        self.taxes_paths = np.zeros((self.n_sims, n_full))
        self.magi_paths = np.zeros((self.n_sims, n_full))
        self.base_spending_paths = np.zeros((self.n_sims, n_full))
        self.lumpy_spending_paths = np.zeros((self.n_sims, n_full))
        self.ssbenefit_paths = np.zeros((self.n_sims, n_full))
        self.portfolio_withdrawal_paths = np.zeros((self.n_sims, n_full))
        self.trust_income_paths = np.zeros((self.n_sims, n_full))
        self.rmd_paths = np.zeros((self.n_sims, n_full))
        self.pension_paths = np.zeros((self.n_sims, n_full))
        self.medicare_paths = np.zeros((self.n_sims, n_full))

        # Set initial balances (unchanged logic)
        for sim_idx in range(self.n_sims):
            self.portfolio_paths[sim_idx, 0] = sum(acct["balance"] for acct in accounts.values())
            for name, acct in accounts.items():
                self.account_paths[name][sim_idx, 0] = acct["balance"]

    def _generate_market_returns(self, n_full):
        """Wrapper for generate_returns to match extraction."""
        return self.generate_returns(n_full + 2)  # Adjust for pre-years if needed; logic unchanged

    def _run_single_simulation(
        self, sim_idx, equity_r_q, bond_r_q, infl_q,
        tax_strategy, irmaa_strategy, max_roth,
        base_annual_spending, withdrawal_rate, travel, gifting,
        accounts, debug_log
    ):
        """Run a single Monte Carlo path (extracted from outer loop)."""
        # 1. Setup (unchanged)
        rng = np.random.default_rng(seed=sim_idx)
        accounts_bal = {name: acct["balance"] for name, acct in accounts.items()}

        # Loop over years
        for year_idx in range(2, self.n_years + 2):  # Start from year 2026 (idx 2)
            year = self.years[year_idx]

            # Extracted: Calculate inflation and portfolio returns
            inflation_this_year, portfolio_return = self._calculate_inflation_and_returns(
                sim_idx, year_idx, equity_r_q, bond_r_q, infl_q, accounts_bal
            )

            # Extracted: Calculate incomes (pension, SS, trust, deferred, RMD)
            pension, ss_benefit, trust_income, deferred_salary, rmd = self._calculate_incomes(
                year, year_idx, accounts_bal, accounts, inflation_this_year
            )

            # Update paths (unchanged)
            self.pension_paths[sim_idx, year_idx] = pension
            self.ssbenefit_paths[sim_idx, year_idx] = ss_benefit
            self.trust_income_paths[sim_idx, year_idx] = trust_income
            self.rmd_paths[sim_idx, year_idx] = rmd

            # Extracted: Calculate spending (base, lumpy, plan)
            base_spending, lumpy_spending, spend_plan = self._calculate_spending(
                sim_idx, year_idx, base_annual_spending, withdrawal_rate, inflation_this_year, rng
            )

            # Update paths (unchanged)
            self.base_spending_paths[sim_idx, year_idx] = base_spending
            self.lumpy_spending_paths[sim_idx, year_idx] = lumpy_spending
            self.plan_paths[sim_idx, year_idx] = spend_plan

            # Extracted: Handle essential withdrawals and tax estimation
            portfolio_draws, final_ordinary, final_ltcg, taxable_ss_benefit, total_estimated_tax = self._handle_essential_withdrawals(
                spend_plan, accounts, accounts_bal, inflation_this_year, ss_benefit
            )

            # Extracted: Handle Roth conversions
            conversion, final_ordinary = self._handle_roth_conversions(
                year, year_idx, sim_idx, accounts_bal, accounts, tax_strategy, irmaa_strategy,
                max_roth, inflation_this_year, final_ordinary, final_ltcg, taxable_ss_benefit
            )

            # Update conversion paths (unchanged)
            self.conversion_paths[sim_idx, year_idx] = conversion

            # Extracted: Handle travel spending
            travel_actual, final_ordinary, final_ltcg, portfolio_draws = self._handle_travel(
                travel=travel,
                year=year,
                inflation_this_year=inflation_this_year,
                final_ordinary=final_ordinary,
                final_ltcg=final_ltcg,
                taxable_ss_benefit=taxable_ss_benefit,
                accounts=accounts,
                accounts_bal=accounts_bal,
                spend_plan=spend_plan,
                portfolio_draws=portfolio_draws,
            )

            # Update travel paths (assuming travel_paths exists; adjust if needed)
            self.travel_paths[sim_idx, year_idx] = travel_actual  # From config or global

            # Extracted: Handle gifting
            gifting_actual, final_ordinary, final_ltcg, portfolio_draws = self._handle_gifting(
                gifting=gifting,
                year=year,
                inflation_this_year=inflation_this_year,
                final_ordinary=final_ordinary,
                final_ltcg=final_ltcg,
                taxable_ss_benefit = taxable_ss_benefit, 
                accounts=accounts,
                accounts_bal=accounts_bal,
                spend_plan=spend_plan,
                portfolio_draws=portfolio_draws,
           )

            # Update gifting paths
            self.gifting_paths[sim_idx, year_idx] = gifting_actual

            # Extracted: Update MAGI, taxes, and adjust for tax differences
            total_tax, medicare = self._update_magi_and_taxes(
                sim_idx, year_idx, final_ordinary, final_ltcg, taxable_ss_benefit,
                ss_benefit, portfolio_draws, total_estimated_tax, accounts_bal, accounts
            )

            # Update paths
            self.taxes_paths[sim_idx, year_idx] = total_tax
            self.medicare_paths[sim_idx, year_idx] = medicare
            self.magi_paths[sim_idx, year_idx] = final_ordinary + final_ltcg + taxable_ss_benefit

            # Fill portfolio and account paths for next year (unchanged)
            self.portfolio_paths[sim_idx, year_idx] = sum(accounts_bal.values())
            for name in accounts_bal.keys():
                self.account_paths[name][sim_idx, year_idx] = accounts_bal[name]
            self.portfolio_withdrawal_paths[sim_idx, year_idx] = portfolio_draws

    def _calculate_inflation_and_returns(self, sim_idx, year_idx, equity_r_q, bond_r_q, infl_q, accounts_bal):
        """Extracted: Calculate inflation and apply portfolio returns."""
        # Quarterly inflation and returns (unchanged)
        q_start = (year_idx - 2) * 4  # Since year_idx starts at 2 for 2026
        q_infl = infl_q[sim_idx, q_start:q_start + 4]
        inflation_this_year = np.prod(1 + q_infl) - 1

        # Apply returns to accounts (unchanged)
        portfolio_return = 0.0
        total_balance = sum(accounts_bal.values())
        if total_balance > 0:
            for acct_name, bal in accounts_bal.items():
                if bal <= 0:
                    continue
                weight = bal / total_balance
                eq_alloc = accounts[acct_name].get("equity_alloc", 0.6)
                bo_alloc = 1 - eq_alloc
                q_eq = equity_r_q[sim_idx, q_start:q_start + 4]
                q_bo = bond_r_q[sim_idx, q_start:q_start + 4]
                acct_return = np.prod(1 + eq_alloc * q_eq + bo_alloc * q_bo) - 1
                accounts_bal[acct_name] *= (1 + acct_return)
                portfolio_return += weight * acct_return

        return inflation_this_year, portfolio_return

    def _calculate_incomes(self, year, year_idx, accounts_bal, accounts, inflation_this_year):
        """Extracted: Calculate all income sources."""
        # Pension (unchanged)
        pension = self.pension_amount * inflation_this_year if self.ages_person1[year_idx] >= self.pension_age else 0

        # Social Security (unchanged)
        ss_benefit = 0
        if self.ages_person1[year_idx] >= self.ss_age_person1:
            ss_benefit += self.ss_amount_person1 * inflation_this_year
        if self.ages_person2[year_idx] >= self.ss_age_person2:
            ss_benefit += self.ss_amount_person2 * inflation_this_year

        # Trust income (unchanged)
        trust_income = 0
        for acct_name, acct in accounts.items():
            if acct.get("tax") == "trust":
                trust_income += acct.get("income", 0) * inflation_this_year

        # Deferred salary (unchanged)
        deferred_salary = self.defered_salary_map.get(year, 0)

        # RMDs (unchanged)
        rmd = 0
        for acct_name, acct in accounts.items():
            if acct.get("rmd_eligible", False):
                age = self.ages_person1[year_idx] if acct["owner"] == "person1" else self.ages_person2[year_idx]
                factor = get_rmd_factor(age)
                if factor > 0:
                    rmd_amount = accounts_bal[acct_name] / factor
                    rmd += rmd_amount
                    accounts_bal[acct_name] -= rmd_amount  # Withdraw RMD

        return pension, ss_benefit, trust_income, deferred_salary, rmd

    def _calculate_spending(self, sim_idx, year_idx, base_annual_spending, withdrawal_rate, inflation_this_year, rng):
        """Extracted: Calculate base, lumpy, and total spending plan."""
        # Base spending (unchanged)
        base_spending = base_annual_spending * inflation_this_year

        # Lumpy spending (home repairs, etc.) (unchanged)
        lumpy_spending = self.sample_home_repair_cost(rng)

        # Total plan (unchanged)
        spend_plan = base_spending + lumpy_spending

        return base_spending, lumpy_spending, spend_plan

    def _handle_essential_withdrawals(self, spend_plan, accounts, accounts_bal, inflation_this_year, ss_benefit):
        """Extracted: Handle withdrawals for essentials and estimate taxes."""
        # Initial draws (unchanged)
        portfolio_draws = max(0, spend_plan - ss_benefit)  # Placeholder

        # Tax estimation (unchanged)
        ordinary_income, ltcg_income = self.estimate_taxable_gap(portfolio_draws, accounts, accounts_bal, inflation_this_year)
        taxable_income_estimate = ordinary_income + ltcg_income
        taxable_ss_benefit = min(0.85 * ss_benefit, max(0, taxable_income_estimate + 0.5 * ss_benefit - 32000) * 0.85)  # Simplified MFJ
        total_estimated_tax = taxable_income_estimate * 0.24  # Rough estimate

        # Adjust plan (unchanged)
        spend_plan += total_estimated_tax
        portfolio_draws = max(0, spend_plan - ss_benefit)

        # Actual withdrawals (unchanged)
        remaining = portfolio_draws
        withdrawal_order = ["taxable", "trust", "inherited", "traditional", "roth"]
        final_ordinary = 0.0
        final_ltcg = 0.0
        for acct_type in withdrawal_order:
            for acct_name, acct in accounts.items():
                if acct["tax"] != acct_type or remaining <= 0:
                    continue
                withdraw_amt = min(accounts_bal[acct_name], remaining)
                accounts_bal[acct_name] -= withdraw_amt
                if acct_type == "taxable":
                    current_gain = accounts_bal[acct_name] - acct["basis"]  # Note: basis not updated here
                    if current_gain > 0:
                        gain_pct = current_gain / accounts_bal[acct_name]
                        realized_gains = withdraw_amt * gain_pct
                        ordinary_part = realized_gains * acct.get("ordinary_pct", 0.1)
                        ltcg_part = realized_gains - ordinary_part
                        final_ordinary += ordinary_part
                        final_ltcg += ltcg_part
                elif acct_type in ["traditional", "inherited"]:
                    final_ordinary += withdraw_amt
                remaining -= withdraw_amt

        return portfolio_draws, final_ordinary, final_ltcg, taxable_ss_benefit, total_estimated_tax

    def _handle_roth_conversions(
        self, year, year_idx, sim_idx, accounts_bal, accounts, tax_strategy, irmaa_strategy,
        max_roth, inflation_this_year, final_ordinary, final_ltcg, taxable_ss_benefit
    ):
        """Extracted: Handle Roth conversions."""
        # Get brackets (unchanged)
        brackets, brackets_dict, thresholds = get_current_brackets(year, inflation_this_year)

        # MAGI estimate (unchanged)
        magi_estimate = final_ordinary + final_ltcg + taxable_ss_benefit

        # Conversion (unchanged)
        traditional_balance = sum(accounts_bal[name] for name, acct in accounts.items() if acct["tax"] == "traditional")
        conversion = optimal_roth_conversion(
            year, traditional_balance, magi_estimate, strategy=tax_strategy,
            tier=irmaa_strategy, max_conversion=max_roth, inflation_this_year=inflation_this_year
        )

        # Apply conversion (unchanged)
        if conversion > 0:
            # Withdraw from traditional
            remaining_conv = conversion
            for name, acct in accounts.items():
                if acct["tax"] == "traditional" and remaining_conv > 0:
                    move = min(accounts_bal[name], remaining_conv)
                    accounts_bal[name] -= move
                    remaining_conv -= move
            # Deposit to Roth
            roth_accounts = self.get_roth_accounts(accounts, "person1")  # Assuming person1; adjust if needed
            for name in roth_accounts:
                accounts_bal[name] += conversion
                break  # Deposit to first Roth

        final_ordinary += conversion

        return conversion, final_ordinary


    def _handle_travel(
        self, travel, year, inflation_this_year,
        final_ordinary, final_ltcg, taxable_ss_benefit,
        accounts, accounts_bal, spend_plan, portfolio_draws
    ):
        """Smart travel: fill remaining tax/IRMAA headroom"""
        leftover = max(0, spend_plan - portfolio_draws)
        target_travel = travel * inflation_this_year
        proposed_travel = min(leftover, target_travel)
        if proposed_travel <= 0:
            return 0, final_ordinary, final_ltcg, portfolio_draws

        # Simulate tax impact
        sim_bal = accounts_bal.copy()
        ord_add, ltcg_add = self.estimate_taxable_gap(proposed_travel, accounts, sim_bal, inflation_this_year)
        projected_magi = final_ordinary + ord_add + ltcg_add + taxable_ss_benefit

        brackets, _, irmaa_thresholds = get_current_brackets(year, inflation_this_year)

        # --- FIXED: Extract HIGH ends of brackets (24%, 32%, etc.) ---
        higher_brackets = []
        for bracket in brackets[3:]:  # Start at 24% bracket
            # bracket = (low, high, rate) — we want HIGH end
            high_end = bracket[1] if np.isfinite(bracket[1]) else 1e12
            if high_end > (final_ordinary + ltcg_add):
                higher_brackets.append(float(high_end))

        next_tax_limit = min(higher_brackets) if higher_brackets else 1e12

        # --- FIXED: Same for IRMAA ---
        higher_irmaa = [float(t) for t in irmaa_thresholds[2:] if t > projected_magi]
        next_irmaa_limit = min(higher_irmaa) if higher_irmaa else 1e12

        headroom = min(next_tax_limit - (final_ordinary + ltcg_add), next_irmaa_limit - projected_magi)

        if headroom < 0:
            reduction = (-headroom) / (1 + 0.30)
            proposed_travel = max(0, proposed_travel + headroom - reduction)

        travel_actual = max(0, int(proposed_travel // 1000) * 1000)

        if travel_actual > 0:
            remaining = travel_actual
            for acct_type in ["taxable", "trust", "inherited", "traditional", "roth"]:
                for name in accounts:
                    if accounts[name]["tax"] != acct_type or remaining <= 0:
                        continue
                    draw = min(accounts_bal[name], remaining)
                    accounts_bal[name] -= draw
                    portfolio_draws += draw
                    remaining -= draw
                if remaining <= 0:
                    break

            ord_real, ltcg_real = self.estimate_taxable_gap(travel_actual, accounts, accounts_bal, inflation_this_year)
            final_ordinary += ord_real
            final_ltcg += ltcg_real

        return travel_actual, final_ordinary, final_ltcg, portfolio_draws

    def _handle_gifting(
        self, gifting, year, inflation_this_year,
        final_ordinary, final_ltcg, taxable_ss_benefit,
        accounts, accounts_bal, spend_plan, portfolio_draws
    ):
        """Smart gifting: use leftover after essentials + travel"""
        leftover = max(0, spend_plan - portfolio_draws)
        target_gifting = gifting * inflation_this_year
        proposed_gifting = min(leftover, target_gifting)
        if proposed_gifting <= 0:
            return 0, final_ordinary, final_ltcg, portfolio_draws

        sim_bal = accounts_bal.copy()
        ord_add, ltcg_add = self.estimate_taxable_gap(proposed_gifting, accounts, sim_bal, inflation_this_year)
        projected_magi = final_ordinary + ord_add + ltcg_add + taxable_ss_benefit

        brackets, _, irmaa_thresholds = get_current_brackets(year, inflation_this_year)

        # --- FIXED: Extract HIGH ends of brackets ---
        higher_brackets = []
        for bracket in brackets[3:]:
            high_end = bracket[1] if np.isfinite(bracket[1]) else 1e12
            if high_end > (final_ordinary + ltcg_add):
                higher_brackets.append(float(high_end))

        next_tax_limit = min(higher_brackets) if higher_brackets else 1e12

        higher_irmaa = [float(t) for t in irmaa_thresholds[2:] if t > projected_magi]
        next_irmaa_limit = min(higher_irmaa) if higher_irmaa else 1e12

        headroom = min(next_tax_limit - (final_ordinary + ltcg_add), next_irmaa_limit - projected_magi)

        if headroom < 0:
            reduction = (-headroom) / (1 + 0.30)
            proposed_gifting = max(0, proposed_gifting + headroom - reduction)

        gifting_actual = max(0, math.ceil(proposed_gifting / 3000) * 3000)

        if gifting_actual > 0:
            remaining = gifting_actual
            trust_draw = roth_draw = 0.0
            for acct_type in ["taxable", "trust", "inherited", "traditional", "roth"]:
                for name in accounts:
                    if accounts[name]["tax"] != acct_type or remaining <= 0:
                        continue
                    draw = min(accounts_bal[name], remaining)
                    accounts_bal[name] -= draw
                    portfolio_draws += draw
                    if accounts[name]["tax"] == "trust":
                        trust_draw += draw
                    if accounts[name]["tax"] == "roth":
                        roth_draw += draw
                    remaining -= draw
                if remaining <= 0:
                    break

            taxable_portion = gifting_actual - trust_draw - roth_draw
            ord_real, ltcg_real = self.estimate_taxable_gap(taxable_portion, accounts, accounts_bal, inflation_this_year)
            final_ordinary += ord_real
            final_ltcg += ltcg_real

        return gifting_actual, final_ordinary, final_ltcg, portfolio_draws

    
    
    def _update_magi_and_taxes(
        self, sim_idx, year_idx, final_ordinary, final_ltcg, taxable_ss_benefit,
        ss_benefit, portfolio_draws, total_estimated_tax, accounts_bal, accounts
    ):
        """Extracted: Update MAGI, calculate taxes, and adjust for differences."""
        magi = final_ordinary + final_ltcg + taxable_ss_benefit

        tax_result = calculate_taxes(
            ordinary_income=final_ordinary,
            ss_benefit=ss_benefit,
            lt_cap_gains=final_ltcg, 
            qualified_dividends=0.0,
            filing_status="married_joint",
            age1=self.ages_person1[year_idx],
            age2=self.ages_person2[year_idx],
            magi_two_years_ago=self.magi_paths[sim_idx, year_idx - 2],
            itemized_deductions=0,
        )
        federal_tax = tax_result["federal_tax"]
        medicare = tax_result["total_medicare"]
        state_tax = tax_result["state_tax_va"]
        total_tax = federal_tax + state_tax
        
        # Adjust for tax differences (unchanged)
        if total_tax > total_estimated_tax:
            shortfall = total_tax - total_estimated_tax
            remaining = shortfall
            portfolio_draws += shortfall
            withdrawal_order = ["taxable", "trust", "inherited", "traditional", "roth"]
            for acct_type in withdrawal_order:
                for acct_name, acct in accounts.items():
                    if acct["tax"] != acct_type or remaining <= 0:
                        continue
                    withdraw_amt = min(accounts_bal[acct_name], remaining)
                    accounts_bal[acct_name] -= withdraw_amt
                    remaining -= withdraw_amt
        else:
            excess = total_estimated_tax - total_tax
            portfolio_draws -= excess
            for acct_name in accounts_bal:
                accounts_bal[acct_name] += excess
                break

        return total_tax, medicare
