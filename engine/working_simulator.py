import numpy as np
import pandas as pd
import copy
import math
from config.user_input import *
from engine.rmd_tables import get_rmd_factor
from engine.tax_utils import get_current_brackets
from engine.roth_optimizer import optimal_roth_conversion
from engine.tax_engine import calculate_taxes


class RetirementSimulator:
    def __init__(self, n_sims=1000):
        self.n_sims = n_sims
        self.n_years = n_years
        self.years = np.arange(current_year - 2, current_year + n_years) # need arrays filled back 2 years for IRMAA
        self.ages_JEF = np.arange(current_age_JEF - 2, current_age_JEF + n_years) # current age is in 2026, array starts 2024
        self.ages_SEF = np.arange(current_age_SEF - 2, current_age_SEF + n_years) # current age is in 2026, array starts 2024
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
        self.deferred_salary_map = {item["year"]: item["amount"] for item in deferred_salary}

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

        # Initialize arrays for trajectories
        sim_years = self.years[2:] # years starts at 2024 for MAGI.  adjust to 2026 onwards
        n_years_sim = len(sim_years)

        account_names = list(accounts.keys())

        n_full = n_years_sim + 2 # need two years prior MAGI for IRMAA
        portfolio_paths = np.zeros((self.n_sims, n_full))
        account_paths = {name: np.zeros((self.n_sims, n_full)) for name in account_names}

        magi_paths = np.zeros((self.n_sims, n_full)) #need to start MAGI 2 years early

        portfolio_end = np.zeros(self.n_sims)
        conversion_paths = np.zeros((self.n_sims, n_full))
        travel_paths = np.zeros((self.n_sims, n_full))
        gifting_paths = np.zeros((self.n_sims, n_full))
        base_spending_paths = np.zeros((self.n_sims, n_full))
        lumpy_spending_paths = np.zeros((self.n_sims, n_full))
        rmd_withdrawal = {name: 0.0 for name in account_names}
        plan_paths = np.zeros((self.n_sims, n_full))
        taxes_paths = np.zeros((self.n_sims, n_full))
        trust_income_paths = np.zeros((self.n_sims, n_full))
        ssbenefit_paths = np.zeros((self.n_sims, n_full))
        portfolio_withdrawal_paths = np.zeros((self.n_sims, n_full))
        rmd_paths = np.zeros((self.n_sims, n_full))
        pension_paths = np.zeros((self.n_sims, n_full))
        medicare_paths = np.zeros((self.n_sims, n_full))
        
        # this generates a full 3D aray for all quarters and all simulations
        equity_r_q, bond_r_q, infl_q = self.generate_returns(n_full)
    
        # --- Main Monte Carlo loop ---
        for sim_idx in range(self.n_sims):
            # create random number for major home repairs
            rng = np.random.default_rng(seed=42 + sim_idx)  # or pass one global seeded rng
            
            accounts_bal = {name: acct["balance"] for name, acct in accounts.items()}

            inflation_index = [1.0] * (n_full + 1) # initialize array

            # Load ending FY25 balance into Portfolio Paths index 0, 1 (right for 2025, approx for 2024)
            # 1. Fill each individual account’s path for y=1 (2025)
            for name in account_names:
                account_paths[name][sim_idx,0] = accounts_bal[name]
                account_paths[name][sim_idx,1] = accounts_bal[name]

            # 2. Fill total portfolio end of year balances for prior 2 years (2024-2025)
            starting_balance = sum(accounts_bal.values()) # from user_inputs
            num = len(accounts_bal)
            portfolio_paths[sim_idx, 0] = 10_300_000 # 2024 number estimate from Vanguard and TIAA-CREF
            portfolio_paths[sim_idx, 1] = starting_balance # from user_inputs
            
            #debug_log.append(f"Starting Balance = ${starting_balance:,.0f} sim_idx = sim_idx \n")

            # 3. Initialize MAGI for 2024, 2025
            magi_paths[sim_idx,0] = 366_069 # actual for 2024
            magi_paths[sim_idx,1] = 335_000 # my current estimate for 2025
 
            # --- Initialize quarter index at FY2026Q1 = quarter_index = 8 ---
            quarter_index = 8

            # --- LOOP OVER YEARS ---
            for y, year in enumerate(range(current_year, current_year + n_years_sim)):
                # arrays start at 2024 and we want simulation to start at currect year (2026)
                year_idx = y + 2
                
                # ------------------------------
                # Calculate inflation index (cummulative) to use throughout
                # ------------------------------
                annual_infl_r = np.prod(1 + infl_q[sim_idx,quarter_index:quarter_index+4]) - 1
                inflation_index[year_idx] = inflation_index[year_idx-1] * (1 + annual_infl_r)
                inflation_this_year = inflation_index[year_idx]     
             
                #debug_log.append(f"annual_infl_r = {annual_infl_r:,.3f} inflation_index =  {inflation_index[year_idx]:,.3f} year_index = {year_idx} \n")
                
                #----------------------
                # Record what the values of required RMDs are for each  account - funds will come out of accounts quarterly
                #----------------------
                rmds = 0.0
                rmd_withdrawal = {}
                
                for acct_name, acct in accounts.items():
                    acct_balance = accounts_bal[acct_name]

                    # Inherited
                    if acct["tax"] == "inherited":
                        rmd_factor = get_rmd_factor(self.ages_JEF[year_idx], 1965, True, True) if acct["owner"] == "JEF" else get_rmd_factor(self.ages_SEF[year_idx], 1959, True, True)
                        rmd_amount = acct_balance / rmd_factor if rmd_factor > 0 else 0
                        rmd_withdrawal[acct_name] = rmd_amount
                        mand = acct_balance / rmd_factor if rmd_factor > 0 else 0
                        rmds += mand

                    # Traditional
                    if acct["tax"] == "traditional":
                        rmd_factor = get_rmd_factor(self.ages_JEF[year_idx], 1965, False, False) if acct["owner"] == "JEF" else get_rmd_factor(self.ages_SEF[year_idx], 1959, False, False)
                        rmd_amount = acct_balance / rmd_factor if rmd_factor > 0 else 0
                        rmd_withdrawal[acct_name] = rmd_amount
                        mand = acct_balance / rmd_factor if rmd_factor > 0 else 0
                        rmds += mand
                rmd_paths[sim_idx,year_idx] = rmds
                 
                # -------------------------------
                # First calculate income, expenses, estimated taxes, ROTH conversions and discretionary sending for the full year
                # ROTH conversions, gifting will be adjusted and taken in Q4 along with final tax calculations
                # This section leaves all account balances intact - those will be adjusted quarterly
                # -------------------------------

                # -------------------------------
                # 1️⃣ Initialize income variables
                # -------------------------------
                trust_income = 0.0
                pension_income = 0.0
                taxable_income_base = 0.0 
                ordinary_income_base = 0.0
                trust_principle_draw = 0.0
                roth_draw = 0.0
                cash_needed = 0.0
                spending_ord_withdrawn = 0.0
                spending_ltcg_withdrawn = 0.0
                gifting_ord_withdrawn = 0.0
                gifting_ord_ltcg = 0.0
                estimated_tax = 0.0
                portfolio_start = portfolio_paths[sim_idx,year_idx-1] #start year with ending balance of prior year
                
                # capture nominal spending plan, e.g. 5% of prior year portfolio
                spend_plan = withdrawal_rate * portfolio_start
                #debug_log.append(f"withdrawal rate = {withdrawal_rate} portfolio_start = {portfolio_start} \n")
                plan_paths[sim_idx,year_idx] = spend_plan

                # do some two-column accounting checks as we move through the code for debugging
                # ONLY add or subtract when funds are generated (income, portfolio withdrawals) or expended
                # do not "spend" for taxes until quarterly tax estimates are determined after ROTH conversions
                cash = 0.0
                spending = 0.0
                regular_income = 0.0
               
                # -------------------------------
                # 2️⃣ Pension, deferred income and Social Security (Annual)
                # -------------------------------
                # Deferred Salary and Pension
                # Income streams
                deferred_income = self.deferred_salary_map.get(current_year,0)
                
                if self.ages_JEF[year_idx] >= pension["JEF"]["start_age"]:
                    pension_income = pension["JEF"]["annual_amount"]
                pension_paths[sim_idx,year_idx] = deferred_income + pension_income

                cash += deferred_income
                cash += pension_income
                regular_income = deferred_income + pension_income
                
                # Social Security
                JEF_SS = 0.0
                SEF_SS = 0.0
                inflation_index_start_jef = 1.0
                inflation_index_start_sef = 1.0
                if self.ages_JEF[year_idx] >= social_security["JEF"]["start_age"]:
                    if year == social_security["JEF"]["start_age"]:
                        inflation_index_start_jef = inflation_index[year_idx]
                    cola_multiplier = inflation_index[year_idx] / inflation_index_start_jef
                    JEF_SS = social_security["JEF"]["annual_pia"] * cola_multiplier
                if self.ages_SEF[year_idx] >= social_security["SEF"]["start_age"]:
                    if year == social_security["SEF"]["start_age"]:
                        inflation_index_start_sef = inflation_index[year_idx]
                    cola_multiplier = inflation_index[year_idx] / inflation_index_start_sef
                    SEF_own = social_security["SEF"]["annual_pia"] * cola_multiplier
                    SEF_SS = max(SEF_own, 0.5 * JEF_SS)

                ss_benefit = JEF_SS + SEF_SS
                ssbenefit_paths[sim_idx,year_idx] = ss_benefit

                cash += ss_benefit
                regular_income += ss_benefit
                
                # Taxable portion of SS
                taxable_ss_benefit = 0.85 * ss_benefit

                # -------------------------------
                # 3️⃣ Trust Income (Annual)
                # -------------------------------
                trust_withdrawal = {}
                for acct_name, acct in accounts.items():
                    acct_balance = accounts_bal[acct_name]

                    # Trust
                    if acct["tax"] == "trust" and acct.get("mandatory_yield", 0) > 0:
                        mand = acct_balance * acct["mandatory_yield"]
                        trust_income += mand
                        trust_withdrawal[acct_name] = mand
                        
                trust_income_paths[sim_idx,year_idx] = trust_income

                cash += trust_income
                regular_income += trust_income
                
                # -------------------------------
                # 4️⃣ Base + lumpy expenses (Annual)
                # -------------------------------
                base_spend = base_annual_spending * inflation_this_year

                #debug_log.append(f"base_spend = {base_spend:,.0f}   base_annual_spending = ${base_annual_spending:,.0f} inflation_this_year = {inflation_this_year:,.3f} \n")

                mortgage = (mortgage_monthly_until_payoff + taxes_and_insurance) * inflation_this_year if year <= mortgage_payoff_year else taxes_and_insurance * inflation_this_year
                # include home costs (taxes, insurance, mortgage while we have one) in base expenses
                base_spend += mortgage

                #debug_log.append(f"base_spend = {base_spend:,.0f} mortgage = {mortgage:,.0f} inflation = {inflation_this_year:,.3f} \n")

                # increase car replacement cycle after 2045 (ages 80 and 86)
                cycle = car_replacement_cycle if year <= 2045 else car_replacement_cycle * 1.5
                car = car_cost_today * ((1 + car_inflation) ** (year - 2025)) if (year - 2025) % cycle == 0 else 0
                home_repair_cost = self.sample_home_repair_cost(rng) * inflation_this_year
                lumpy_expenses = car + home_repair_cost

                base_spending_paths[sim_idx,year_idx] = base_spend
                lumpy_spending_paths[sim_idx,year_idx] = lumpy_expenses

                essential_spending = base_spend + lumpy_expenses
                
                spending += essential_spending
                
                # -------------------------------
                # 5️⃣ Calulate taxable income estimate for portfolio draw for essential spending and associated taxes
                # -------------------------------
                taxable_income_base = rmds + pension_income + taxable_ss_benefit + trust_income
                ordinary_income_base = rmds + pension_income + trust_income

                # add portfolio withdrawal that will cover gap to ordinary income + SS benefits
                portfolio_draw_for_expenses = max(0, essential_spending - (ordinary_income_base + ss_benefit))
        
                # Compute taxable income on gap income needed

                # This routine estimates what part is ordinary income vs LTCG income (NOT TAXES)
                tax_sim_bal = accounts_bal.copy()   # copy so doesn't modify real balances
                ordinary_income_essentials, ltcg_income_essentials = self.estimate_taxable_gap(
                    cash_needed=portfolio_draw_for_expenses,
                    accounts=accounts,
                    accounts_bal=tax_sim_bal,
                    inflation_this_year=inflation_this_year
                )

                # This calculates taxes due on income
                tax_result = calculate_taxes(
                    ordinary_income=ordinary_income_base + ordinary_income_essentials,
                    ss_benefit=ss_benefit,
                    lt_cap_gains=ltcg_income_essentials,
                    qualified_dividends=0.0,
                    filing_status="married_joint",
                    age1=self.ages_JEF[year_idx],
                    age2=self.ages_SEF[year_idx],
                    magi_two_years_ago=magi_paths[sim_idx,year_idx-2],# note that 2024, 2025 were added at front of array
                    itemized_deductions=0,
                )
                federal_tax = tax_result["federal_tax"]
                medicare = tax_result["total_medicare"]
                state_tax = tax_result["state_tax_va"]

                # Need to take portfolio draw to pay estimated taxes, so iterate again
                # add portfolio withdrawal that will cover gap to ordinary income + SS benefits to pay taxes
                taxes = federal_tax + state_tax
                
                # if no draw beyond base income yet, check if base income (RMDs, Trust Income, Pension, SS) can cover taxes
                if portfolio_draw_for_expenses == 0.0:
                    portfolio_draw_for_taxes = max(0, essential_spending + taxes - (ordinary_income_base + ss_benefit))
                else:
                    portfolio_draw_for_taxes = taxes
                
                #debug_log.append(f"taxes = ${taxes:,.0f} draw for taxes = ${portfolio_draw_for_taxes:,.0f} \n")

                # This routine estimates what part is ordinary income vs LTCG income (NOT TAXES)
                tax_sim_bal = accounts_bal.copy()   # copy so doesn't modify real balances
                ordinary_income_for_taxes, ltcg_income_for_taxes = self.estimate_taxable_gap(
                    cash_needed=portfolio_draw_for_taxes,
                    accounts=accounts,
                    accounts_bal=tax_sim_bal,
                    inflation_this_year=inflation_this_year
                )

                # This calculates taxes due on income
                tax_result = calculate_taxes(
                    ordinary_income=ordinary_income_base + ordinary_income_essentials + ordinary_income_for_taxes,
                    ss_benefit=ss_benefit,
                    lt_cap_gains=ltcg_income_essentials + ltcg_income_for_taxes,
                    qualified_dividends=0.0,
                    filing_status="married_joint",
                    age1=self.ages_JEF[year_idx],
                    age2=self.ages_SEF[year_idx],
                    magi_two_years_ago=magi_paths[sim_idx,year_idx-2],# note that 2024, 2025 were added at front of array
                    itemized_deductions=0,
                )
                federal_tax = tax_result["federal_tax"]
                medicare = tax_result["total_medicare"]
                state_tax = tax_result["state_tax_va"]
                
                # Discretionary spending (travel, gifting, ROTH conversions) all rely on estimated MAGI and total draw on portfolio
                # being conservative and assuming draws to cover "tax on draws to pay main taxes" are 100% income draw from portfolio
                
                magi_essentials = taxable_income_base + ordinary_income_essentials + ltcg_income_essentials + ordinary_income_for_taxes + ltcg_income_for_taxes
                
                portfolio_draws_essentials = rmds + trust_income + portfolio_draw_for_expenses + portfolio_draw_for_taxes
                
                cash += portfolio_draw_for_taxes
                #debug_log.append(f"Cash Flow after Essentials + taxes on Essentials \n")
                #debug_log.append(f"Cash = ${cash:,.0f}    TAX ESTIMATE = ${federal_tax + state_tax:,.0f} \n")
                
                # -------------------------------
                # 6️⃣ Discretionary Travel
                # -------------------------------
                #
                # spend plan is to cap portfolio withdrawals at 5% of prior year
                # other income (pension, SS) do not count against this cap
                #
                leftover = max(0, spend_plan - portfolio_draws_essentials)

                target_travel = travel if year <= 2035 else travel / 2
                target_travel = target_travel * inflation_this_year
                proposed_travel = min(target_travel, leftover)

                # Now calculate tax basis for portfolio draw to pay for proposed travel
                tax_sim_bal = accounts_bal.copy()   # copy so doesn't modify real balances
                ordinary_income, ltcg_income = self.estimate_taxable_gap(
                    cash_needed=proposed_travel,
                    accounts=accounts,
                    accounts_bal=tax_sim_bal,
                    inflation_this_year=inflation_this_year
                )
                magi_proposed_travel = magi_essentials + ordinary_income + ltcg_income

                # Check that we have not violated tax and IRMAA strategies
                # Determine target Federal Tax Brackets and IRMAA thresholds
                brackets, brackets_dict, thresholds = get_current_brackets(year, inflation_this_year)
                #old syntax brackets, thresholds = get_current_brackets(year, inflation_this_year) 
                fill_targets = {
                    "fill_12_percent": brackets[1][1],
                    "fill_22_percent": brackets[2][1],
                    "fill_24_percent": brackets[3][1],
                    "fill_32_percent": brackets[4][1],
                }
                fill_thresholds = {
                    "fill_IRMAA_1": thresholds[1],
                    "fill_IRMAA_2": thresholds[2],
                    "fill_IRMAA_3": thresholds[3],
                    "fill_IRMAA_4": thresholds[4],
                }
                # Apply tax strategy - fill a tax bracket
                if tax_strategy.startswith("fill_"):
                    bracket = fill_targets.get(tax_strategy)

                # Apply IRMAA threshold if there is one
                if irmaa_strategy.startswith("fill_"):
                    threshold = fill_thresholds.get(irmaa_strategy)

                if magi_proposed_travel > min(bracket, threshold):
                    # Assumes we are in 24% marginal Federal bracket and 5.75% State
                    over = magi_proposed_travel - min(bracket, threshold)
                    proposed_travel = proposed_travel - over / (1 + 0.24 + 0.0575)

                actual_travel = max(0, math.ceil(proposed_travel / 5000) * 5000) # round to next $5000
                travel_paths[sim_idx,year_idx] = actual_travel

                # This routine estimates what part is ordinary income vs LTCG income (NOT TAXES)
                tax_sim_bal = accounts_bal.copy()   # copy so doesn't modify real balances
                travel_ordinary_income, travel_ltcg_income = self.estimate_taxable_gap(
                    cash_needed=actual_travel,
                    accounts=accounts,
                    accounts_bal=tax_sim_bal,
                    inflation_this_year=inflation_this_year
                )

                taxable_income_before_conv = magi_essentials + travel_ordinary_income + travel_ltcg_income
                
                cash += 0.0 # we have not pulled any cash out here
                spending += actual_travel # deal with taxes after ROTH
                #debug_log.append(f"Cash Flow after Travel.   \n")
                #debug_log.append(f"Cash = ${cash:,.0f}    Spending = ${spending:,.0f}   Base Spending = ${base_spend:,.0f}   Lumpy Expenses = ${lumpy_expenses:,.0f} Travel =${actual_travel:,.0f}\n")
                #debug_log.append(f"Year = {year}   Tax Bracket = ${bracket:,.0f}   IRMAA Thersold = ${threshold:,.0f} \n")

                # -------------------------------
                # 6️⃣ Roth conversions (prioritize over Gifting)
                # -------------------------------

                # SEF
                person = "SEF"
                trad_accts = {k: v for k, v in accounts_bal.items() if k.startswith(person) and accounts[k]["tax"] == "traditional"}
                trad_total = sum(trad_accts.values())
                roth_accts = {k: v for k, v in accounts_bal.items() if k.startswith(person) and accounts[k]["tax"] == "roth"}
                roth_name = next(iter(roth_accts.keys()))
                conv_sef = 0.0
                if trad_total > 0:
                    conv_sef = optimal_roth_conversion(year, trad_total, taxable_income_before_conv, tax_strategy, irmaa_strategy, max_roth, inflation_this_year)
  
                # JEF
                person = "JEF"
                trad_accts = {k: v for k, v in accounts_bal.items() if k.startswith(person) and accounts[k]["tax"] == "traditional"}
                trad_total = sum(trad_accts.values())
                roth_accts = {k: v for k, v in accounts_bal.items() if k.startswith(person) and accounts[k]["tax"] == "roth"}
                roth_name = next(iter(roth_accts.keys()))
                conv_jef = 0.0
                if trad_total > 0:
                    conv_jef = optimal_roth_conversion(year, trad_total, taxable_income_before_conv + conv_sef, tax_strategy, irmaa_strategy, max_roth - conv_sef, inflation_this_year)

                conversion_paths[sim_idx,year_idx] = conv_jef + conv_sef
                taxable_income_before_gifting = taxable_income_before_conv + conv_sef + conv_jef

                # Plan to take ROTH conversions in Q4 so we can just deal with tax burden then and not estimated taxes now.
                # Assumes we are in 24% bracket (most likley ROTH strategy)
                #estimated_tax_on_roth = (0.24 + 0.0575) * (conv_sef + conv_jef)
                portfolio_draw_total = portfolio_draw_for_expenses + actual_travel + portfolio_draw_for_taxes #+ estimated_tax_on_roth

                magi_estimate = taxable_income_before_conv + conv_sef + conv_jef
 
                # debug_log.append(f"MAGI after ROTH = ${magi_estimate:,.0f} TAX BRACKET = ${bracket:,.0f} IRMAA THRESHOLD = ${threshold:,.0f} \n")

                # ----------------------------
                # Calculate estimated taxes (to take quarterly withdrawals to pay bills)
                # ----------------------------

                # This routine estimates what part is ordinary income vs LTCG income (NOT TAXES)
                # applying to full portfolio withdrawal
                tax_sim_bal = accounts_bal.copy()   # copy so doesn't modify real balances
                portfolio_draw_ordinary_income, portfolio_draw_ltcg_income = self.estimate_taxable_gap(
                    cash_needed=portfolio_draw_total,
                    accounts=accounts,
                    accounts_bal=tax_sim_bal,
                    inflation_this_year=inflation_this_year
                )

                # gather final sum of ordinary income streams
                ordinary_income = trust_income + rmds + conv_sef + conv_jef + portfolio_draw_ordinary_income
                # This calculates taxes due on income
                tax_result = calculate_taxes(
                    ordinary_income=ordinary_income,
                    ss_benefit=ss_benefit,
                    lt_cap_gains=portfolio_draw_ltcg_income,
                    qualified_dividends=0.0,
                    filing_status="married_joint",
                    age1=self.ages_JEF[year_idx],
                    age2=self.ages_SEF[year_idx],
                    magi_two_years_ago=magi_paths[sim_idx,year_idx-2],# note that 2024, 2025 were added at front of array
                    itemized_deductions=0,
                )
                federal_tax = tax_result["federal_tax"]
                state_tax = tax_result["state_tax_va"]
                total_estimated_tax = federal_tax + state_tax

                #--------------------------------
                # Sum up portfolio draws needed to cover base expenses, travel and estimated taxes
                #--------------------------------
                #
                #*** defintions reminders from eearlier in code to check math/logic at play here***
                #*** portfolio_draw_for_expenses = max(0, essential_spending - (ordinary_income_base + ss_benefit))
                #*** essential_spending = base_spend + lumpy_expenses
                #*** ordinary_income_base = rmds + pension_income + trust_income
                
                #>>> portfolio_draw_for_expenses = base_spend + lumpy_expenses - (rmds + pension_income + trust_income + ss_benefit))
                
                portfolio_draw_add = portfolio_draw_for_expenses + actual_travel + total_estimated_tax
                #*** so above includes all outflows except gifting and all inflows including RMDs and Trust Income
                portfolio_draw_total = rmds + trust_income + portfolio_draw_add

                cash += portfolio_draw_add # Now we are drawing from portfolio ABOVE AND BEYOND RMDS AND TRUST INCOME
                spending += total_estimated_tax # And we are paying estimated taxes
                #debug_log.append(f"Cash Flow after estimated taxes and portfolio draw determined for year - EXCLUDES GIFTING \n")
                #debug_log.append(f"Cash = ${cash:,.0f}    Spending = ${spending:,.0f}   Estimated Taxes = ${total_estimated_tax:,.0f} \n")

                #--------------------------------
                # Now do quarterly withdrawals
                # Then apply investment returns to remaining balance (this presumes funds pulled at start of quarter)
                #--------------------------------

                final_ordinary = 0.0 # used to track actual tax bases through real account draws
                final_ltcg = 0.0
                portfolio_draws = 0.0 #explicitly NOT RMD and Trust Income (to equal portfolio_draw_add when complete)

                for q in range(4):

                    # -------------------------------
                    # Withdraw RMDs
                    # -------------------------------
                    for acct_type in ["inherited", "traditional"]:
                        for acct_name, acct in accounts.items():
                            if acct["tax"] != acct_type:
                                continue
                            acct_balance = accounts_bal[acct_name]
                            withdraw_amt = min(acct_balance, rmd_withdrawal[acct_name] / 4) # divide by 4 for quarterly amount
                            accounts_bal[acct_name] -= withdraw_amt
                            final_ordinary += withdraw_amt
  
                    # -------------------------------
                    # Withdraw Trust Income (use quarterly balances here vs estimates done for taxes above
                    # -------------------------------
                    for acct_type in ["trust"]:
                        for acct_name, acct in accounts.items():
                            if acct["tax"] != acct_type:
                                continue
                            acct_balance = accounts_bal[acct_name]
                            withdraw_amt = trust_withdrawal[acct_name] / 4 # divide by 4 for quarterly amount
                            accounts_bal[acct_name] -= withdraw_amt
                            final_ordinary += withdraw_amt                            
 
                    # -------------------------------
                    # Withdraw from portfolio for essentials, travel and estimated taxes
                    # -------------------------------
                    essential_withdrawn_total = 0.0

                    quarterly_portfolio_withdrawal_need = (portfolio_draw_add) / 4 # RMDs and Trust Income explicitly not in this
                    # start with $50k PER QUARTER from trusts, inflated
                    # this is a draw on principle so it is not taxed
                    trust_remain = min(50000 * inflation_this_year,quarterly_portfolio_withdrawal_need)
                    for acct_type in ["trust"]:
                        for acct_name, acct in accounts.items():
                            if acct["tax"] != acct_type:
                                continue
                            if trust_remain <= 0:
                                break

                            # Determine how much to withdraw
                            acct_balance = accounts_bal[acct_name]
                            withdraw_amt = min(acct_balance, trust_remain)

                            # Apply withdrawal
                            accounts_bal[acct_name] -= withdraw_amt
                            essential_withdrawn_total += withdraw_amt
                            trust_remain -= withdraw_amt
                            final_ordinary += 0 # Trust principle draws are not taxable income                           
                            portfolio_draws += withdraw_amt

                    # Then take the rest in order by accounts        
                    remaining = quarterly_portfolio_withdrawal_need - essential_withdrawn_total

                    # Define withdrawal priority
                    withdrawal_order = [
                        "taxable",          # ordinary first, then LTCG
                        "trust",            # ordinary
                        "inherited",        # ordinary
                        "traditional",      # ordinary
                        "roth"              # tax-free last
                    ]

                    for acct_type in withdrawal_order:
                        for acct_name, acct in accounts.items():
                            if acct["tax"] != acct_type:
                                continue
                            if remaining <= 0:
                                break

                            # Determine how much to withdraw
                            acct_balance = accounts_bal[acct_name]
                            withdraw_amt = min(acct_balance, remaining)

                            # Track taxable portions
                            if acct_type == "taxable":
                                # calculate percentage that is unrealized gains
                                current_gain = acct["balance"] - acct["basis"] # this is fixed at original numbers in use_inputs.  Tracking cost basis woukld be ideal.Then use account_bal and account_bas.
                                if current_gain >0:
                                    gain_percentage = current_gain / acct["balance"]
                                    realized_gains = withdraw_amt * gain_percentage
                                    return_of_basis = withdraw_amt - realized_gains
                                else:
                                    realized_gains = 0.0
                                    return_of_basis = withdraw_net
                                # Split TAXABLE portion[realized_gains] between ordinary vs LTCG
                                ordinary_part = realized_gains * acct.get("ordinary_pct", 0.1) #this assumes a portfolio heavily weighed to stocks (LTCG)
                                ltcg_part = realized_gains - ordinary_part
                                final_ordinary += ordinary_part
                                final_ltcg += ltcg_part
                            elif acct_type == "trust":
                                final_ordinary += 0 # Trust principle draws are not taxable income                           
                            elif acct_type == "roth":
                                final_ordinary += 0 # ROTH draws are not taxable income                           
                            else: # taxable IRA draws
                                final_ordinary += withdraw_amt

                            # Apply withdrawal
                            accounts_bal[acct_name] -= withdraw_amt
                            essential_withdrawn_total += withdraw_amt
                            remaining -= withdraw_amt
                            portfolio_draws += withdraw_amt

                    # debug_log.append(f"Withdrawl: Essentials = ${essential_withdrawn_total:,.0f} \n")

                    # -------------------------------
                    # 1️⃣2️⃣ Apply market returns & update account_paths
                    # -------------------------------
                    for name, acct in accounts.items():
                        eq = acct["equity"]
                        ret = eq * equity_r_q[sim_idx, quarter_index] + (1 - eq) * bond_r_q[sim_idx, quarter_index] #use quarterly returns
                        accounts_bal[name] *= (1 + ret)

                    quarter_index += 1
                    # End of quarterly loop

                    #debug_log.append(f"Now we are through quarterly withdrawal loop - compare what we planned ot take vs took \n")
                    #debug_log.append(f"Plan Portfolio Draw = ${portfolio_draw_add:,.0f}  Actual Portfolio Draws (excludes RMDs and Trust Income) = ${portfolio_draws:,.0f} \n")

                # At this point ROTH conversions and gifting remain to be taken from accounts in Q4
                #-------------------------
                # Execute ROTH conversions in Q4 (move funds from Trad to Roth)
                #-------------------------
                person = "SEF"
                # Re-compute at time of conversion (Q4)
                trad_accts = {k: accounts_bal[k] for k, v in accounts.items()
                if k.startswith("SEF") and v["tax"] == "traditional" and accounts_bal[k] > 0}
                roth_accts = {k: accounts_bal[k] for k, v in accounts.items() 
                if k.startswith("SEF") and v["tax"] == "roth"}
                roth_name = next(iter(roth_accts))  # pick first Roth account
                trad_total = sum(trad_accts.values())
                if trad_total >0 and conv_sef >0:
                    for acct_name, bal in trad_accts.items():
                        fraction = bal / trad_total
                        move = min(conv_sef * fraction, accounts_bal[acct_name])
                        accounts_bal[acct_name] -= move
                        accounts_bal[roth_name] += move
                        final_ordinary += move # ROTH conversions are ordinary income

                person = "JEF"
                trad_accts = {k: accounts_bal[k] for k, v in accounts.items()
                if k.startswith("JEF") and v["tax"] == "traditional" and accounts_bal[k] > 0}
                roth_accts = {k: accounts_bal[k] for k, v in accounts.items() 
                if k.startswith("JEF") and v["tax"] == "roth"}
                roth_name = next(iter(roth_accts))  # pick first Roth account
                trad_total = sum(trad_accts.values())
                if trad_total >0 and conv_jef >0:
                    for acct_name, bal in trad_accts.items():
                        fraction = bal / trad_total
                        move = min(conv_jef * fraction, accounts_bal[acct_name])
                        accounts_bal[acct_name] -= move
                        accounts_bal[roth_name] += move
                        final_ordinary += move # ROTH conversions are ordinary income

                # -------------------------------
                # Re-Compute leftover cash for gifting
                # -------------------------------
                #
                leftover = max(0, spend_plan - portfolio_draws)
                target_gifting =  gifting * inflation_this_year
                proposed_gifting = min(leftover, target_gifting)
                
                # Now calculate tax basis for portfolio withdrawals to pay for proposed gifting
                tax_sim_bal = accounts_bal.copy()   # copy so doesn't modify real balances
                ordinary_income, ltcg_income = self.estimate_taxable_gap(
                    cash_needed=proposed_gifting,
                    accounts=accounts,
                    accounts_bal=tax_sim_bal,
                    inflation_this_year=inflation_this_year
                )
                taxable_income_proposed_gifting = final_ordinary + ordinary_income + ltcg_income # final_ordinary includes essentials, lumpy and travel
                #
                # Check MAGI against tax and IRMAA strategies and adjust leftover_cash accordingly
                #
                if taxable_income_proposed_gifting > min(bracket, threshold):
                    # Assumes we are in 24% marginal Federal bracket and 5.75% State
                    over = taxable_income_proposed_gifting - min(bracket, threshold)
                    proposed_gifting = gifting - over / (1 + 0.24 +0.0575)

                actual_gifting = max(0, math.ceil(proposed_gifting / 3000) * 3000) # round to nearest $3000
                gifting_paths[sim_idx,year_idx] = actual_gifting
                
                # Withdraw gifting from accounts (track taxable portions)
                remaining = actual_gifting
                for acct_type in withdrawal_order:
                    for acct_name, acct in accounts.items():
                        if acct["tax"] != acct_type or remaining <= 0:
                            continue
                        withdraw_amt = min(accounts_bal[acct_name], remaining)
                        accounts_bal[acct_name] -= withdraw_amt
                        portfolio_draws += withdraw_amt
                        if acct["tax"] == "trust":
                            trust_principle_draw = withdraw_amt
                        if acct["tax"] == "roth":
                            roth_draw = withdraw_amt
                        remaining -= withdraw_amt
                        if remaining <= 0:
                            break
                    if remaining <= 0:
                        break

                # Now calculate tax basis for portfolio draw to pay for actual gifting
                # first deduct any portion of gifting funded by trust principle or ROTH
                tax_sim_bal = accounts_bal.copy()   # copy so doesn't modify real balances
                taxable_gifting = actual_gifting - trust_principle_draw - roth_draw
                gifting_ordinary_income, gifting_ltcg_income = self.estimate_taxable_gap(
                    cash_needed=taxable_gifting,
                    accounts=accounts,
                    accounts_bal=tax_sim_bal,
                    inflation_this_year=inflation_this_year
                )
                final_ordinary += gifting_ordinary_income # final_ordinary includes essentials, lumpy and travel
                final_ltcg += gifting_ltcg_income 
 
                # -------------------------------
                # 🔟 Update MAGI & taxes
                # -------------------------------
                #
                magi = final_ordinary + final_ltcg + taxable_ss_benefit

                magi_paths[sim_idx,year_idx] = magi

                tax_result = calculate_taxes(
                    ordinary_income=final_ordinary,
                    ss_benefit=ss_benefit,
                    lt_cap_gains=final_ltcg, 
                    qualified_dividends=0.0,
                    filing_status="married_joint",
                    age1=self.ages_JEF[year_idx],
                    age2=self.ages_SEF[year_idx],
                    magi_two_years_ago=magi_paths[sim_idx,year_idx - 2], # note that 2024, 2025 were added at front of array
                    itemized_deductions=0,
                )
                federal_tax = tax_result["federal_tax"]
                medicare = tax_result["total_medicare"] # This is the Parts B and D surcharge, not an actual tax.  I include that in our base spending nees already
                state_tax = tax_result["state_tax_va"]
                total_tax =  federal_tax + state_tax
                taxes_paths[sim_idx,year_idx] = total_tax
                medicare_paths[sim_idx,year_idx] = medicare
                
                # Need to compare estimated to actual taxes and see if we have a surplus or need more funding for those
                excess = 0.0
                surplus = 0.0
                if total_tax > total_estimated_tax: # then we need to draw funds to cover
                    shortfall = total_tax - total_estimated_tax
                    # Withdraw shortfall from accounts
                    remaining = shortfall
                    portfolio_draws += shortfall #
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
                else: # Put excess funds in first cash account
                    excess = total_estimated_tax - total_tax
                    portfolio_draws -= excess
                    for acct_name, acct in accounts.items():
                        accounts_bal[acct_name] += excess
                        break
                #debug_log.append(f"Year = {year}   Excess = ${excess:,.0f}  surplus = ${surplus:,.0f} \n \n \n")
 
                # fill portfolio_paths array with starting balance for following year
                portfolio_paths[sim_idx,year_idx] = sum(accounts_bal.values())
                for name in account_names:
                    account_paths[name][sim_idx,year_idx] = accounts_bal[name]
                portfolio_withdrawal_paths[sim_idx,year_idx] = portfolio_draws  # Exclusive of RMDs amd Trust Income

            # end of loop over years
            
        # end of loop over sim_idx

        # 7. Stats
        threshold = 500_000
        portfolio_end = portfolio_paths[:, -1]
        success_rate = np.mean(portfolio_end > threshold) * 100
        minimum_annual_balance = np.min(portfolio_paths, axis=1)
        avoid_ruin_rate = np.mean(minimum_annual_balance > threshold) *100
        

        result = {
            "success_rate": success_rate,
            "avoid_ruin_rate": avoid_ruin_rate,
            "median_final": np.median(portfolio_end),
            "p10_final": np.percentile(portfolio_end, 10),
            "account_paths": account_paths,
            "conversion_paths": conversion_paths,
            "travel_paths": travel_paths,
            "gifting_paths": gifting_paths,
            "base_spending_paths": base_spending_paths,
            "lumpy_spending_paths": lumpy_spending_paths,
            "plan_paths": plan_paths,
            "taxes_paths": taxes_paths,
            "magi_paths": magi_paths,
            "trust_income_paths": trust_income_paths,
            "ssbenefit_paths": ssbenefit_paths,
            "portfolio_withdrawal_paths": portfolio_withdrawal_paths,
            "rmd_paths": rmd_paths,
            "pension_paths": pension_paths,
            "medicare_paths": medicare_paths,
            "years": self.years
        }

        if return_trajectories:
            result["portfolio_paths"] = portfolio_paths

        return result
