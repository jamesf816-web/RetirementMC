# Withdrawal_engine.py
#
# Handlkes logic for prioritizing account withdrawals
#
    def _get_withdrawal_order(self):
        """
        Future Proofing: Read this from XML/Inputs later.
        Current: Hardcoded standard order.
        """
        return ["taxable", "trust", "inherited", "traditional", "roth"]

    def _withdraw_from_hierarchy(self, 
                                 cash_needed: float, 
                                 accounts_bal: dict, 
                                 order: list, 
                                 simulate_only: bool = False) -> dict:
        """
        The Core Engine: Withdraws cash_needed following the order list.
        
        Args:
            simulate_only: If True, uses a copy of balances to just estimate taxes/basis.
            
        Returns: 
            Dict containing: 
            {'withdrawn': float, 'ordinary_inc': float, 'ltcg_inc': float, 'balances': dict}
        """
        working_bal = accounts_bal.copy() if simulate_only else accounts_bal
        remaining = cash_needed
        total_withdrawn = 0.0
        ord_inc = 0.0
        ltcg_inc = 0.0
        
        for acct_type in order:
            # Filter accounts by type (preserve original iteration order)
            targets = [k for k, v in self.accounts.items() if v["tax"] == acct_type]
            
            for name in targets:
                if remaining <= 0: break
                
                bal = working_bal[name]
                if bal <= 0: continue
                
                amt = min(bal, remaining)
                
                # Logic Update
                working_bal[name] -= amt
                remaining -= amt
                total_withdrawn += amt
                
                # Tax Characterization
                if acct_type == "taxable":
                    # Replicate your gain logic here
                    acct_ref = self.accounts[name]
                    # Note: To perfectly replicate your logic, you need basis tracking
                    # For now, replicate the logic:
                    curr_gain = (bal + amt) - acct_ref["basis"] # Approximation based on original logic flow
                    # (You may need to pass exact original basis if not modified in place)
                    
                    if (bal) > 0: # Avoid div by zero
                        gain_pct = max(0, (bal - acct_ref.get("basis", bal)) / bal) # Simplified
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

