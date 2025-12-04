from models import PlannerInputs
from config.default_portfolio import accounts as Account
from typing import Dict, Any, List

def get_planner_inputs(portfolio_data: Dict[str, Dict[str, Any]], setup_data: List[Dict[str, Any]]) -> PlannerInputs:
    """
    Adapter function to convert raw dictionary and list data from Dash stores 
    into the structured, typed PlannerInputs object.
    
    This function isolates data mapping logic from the core simulation engine.
    """

    if not setup_data or not isinstance(setup_data, list) or not setup_data[0]:
        raise ValueError("Setup data is empty or improperly structured.")
        
    setup_params = setup_data[0]

    # 1. Map Accounts (list of dictionaries to dictionary of Account objects)
    typed_accounts: Dict[str, Account] = {}
    for name, raw_acct in portfolio_data.items():
        try:
            # Ensure proper type casting for the Account dataclass fields
            typed_accounts[name] = Account(
                balance=float(raw_acct.get("balance", 0)),
                equity=float(raw_acct.get("equity", 0)),
                bond=float(raw_acct.get("bond", 0)),
                tax=raw_acct.get("tax", "traditional"),
                owner=raw_acct.get("owner", "person1"),
                # Explicitly handle optional fields that might be None or empty string
                basis=float(raw_acct["basis"]) if raw_acct.get("basis") not in [None, ""] else None,
                mandatory_yield=float(raw_acct["mandatory_yield"]) if raw_acct.get("mandatory_yield") not in [None, ""] else None,
                rmd_factor_table=raw_acct.get("rmd_factor_table") 
            )
        except (ValueError, TypeError) as e:
            raise TypeError(f"Error mapping account '{name}'. Check data types. Error: {e}")


    # 2. Extract and Map Global Setup Parameters
    try:
        nsims_raw = setup_params.get("nsims", 1000)
        
        # Build the final PlannerInputs object
        inputs = PlannerInputs(
            start_age=int(setup_params.get("current_age", 30)),
            retirement_age=int(setup_params.get("retirement_age", 65)),
            end_age=int(setup_params.get("death_age", 95)),
            initial_spending=float(setup_params.get("annual_spending", 40000)),
            annual_savings=float(setup_params.get("annual_savings", 10000)),
            nsims=int(nsims_raw),
            accounts=typed_accounts
        )
        return inputs
        
    except (ValueError, TypeError) as e:
        raise TypeError(f"Error mapping global setup parameters. Check data types in setup grid. Error: {e}")
