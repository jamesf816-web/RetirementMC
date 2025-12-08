# withdrawal_engine.py

# Handlkes logic for prioritizing account withdrawals
#
class WithdrawalEngine:
    """
    Handles logic for prioritizing account withdrawals based on tax strategy.
    """
    def __init__(self, inputs, accounts_metadata):
        self.inputs = inputs
        self.accounts_metadata = accounts_metadata
        
    def _get_withdrawal_order(self) -> list:
        """
        Dynamically determines the withdrawal hierarchy based on the tax strategy.
        """
        tax_strategy = self.inputs.tax_strategy
        
        # Determine the order of accounts to withdraw from based on strategy
        if tax_strategy == 'maximize_roth':
            # Priority: Taxable -> Traditional/Inherited -> Roth (maximize Roth life)
            return ["taxable", "trust", "traditional", "inherited", "def457b", "roth"]
        elif tax_strategy == 'maximize_traditional':
            # Priority: Roth -> Traditional/Inherited -> Taxable
            return ["roth", "trust", "traditional", "inherited", "def457b", "taxable"]
        else:
            # Default to drawing down tax-deferred first to manage RMDs
            return ["traditional", "def457b", "inherited", "taxable", "roth", "trust"]


    def _withdraw_from_hierarchy(self, 
                                 cash_needed: float, 
                                 accounts_bal: dict, 
                                 simulate_only: bool = False) -> dict:
        """
        The Core Engine: Withdraws cash_needed following the dynamically generated order.
        
        Args:
            cash_needed: The cash needed from the portfolio.
            accounts_bal: The current state of account balances (from the simulation path).
            simulate_only: If True, uses a copy of balances to just estimate taxes/basis.
            
        Returns: 
            Dict containing: 
            {'withdrawn': float, 'ordinary_inc': float, 'ltcg_inc': float, 'balances': dict}
        """
        # Determine the withdrawal order dynamically
        order = self._get_withdrawal_order()

        working_bal = accounts_bal
        if simulate_only:
            working_bal = copy.ddepcopy(accounts_bal)

        remaining = cash_needed
        total_withdrawn = 0.0
        ord_inc = 0.0
        ltcg_inc = 0.0
        
        for acct_type in order:
            # Filter accounts by type (preserve original iteration order)
            targets = [k for k, v in self.accounts_metadata.items() if v["tax"] == acct_type]
            
            for name in targets:
                acct_state = working_bal[name]
                numerical_balance = acct_state.get("balance", 0.0)
                if numerical_balance <= 0: continue
                
                amt = min(numerical_balance, remaining)
                
                acct_state["balance"] -= amt
                remaining -= amt
                total_withdrawn += amt
                
                # Tax Characterization (Uses self.accounts_metadata)
                if acct_type == "taxable":
                    acct_ref = self.accounts_metadata[name]
                    # Note: To perfectly replicate your logic, you need basis tracking
                    if  numerical_balance > 0:
                        gain_pct = max(0, (numerical_balance - acct_ref.get("basis", numerical_balance)) / numerical_balance) 
                        realized = amt * gain_pct
                        ord_part = realized * acct_ref.get("ordinary_pct", 0.1)
                        ltcg_part = realized - ord_part
                        ord_inc += ord_part
                        ltcg_inc += ltcg_part
                elif acct_type in ["traditional", "inherited", "def457b"]:
                    ord_inc += amt
                    
        return {
            "withdrawn": total_withdrawn,
            "ordinary_inc": ord_inc,
            "ltcg_inc": ltcg_inc,
            "balances": working_bal
        }
