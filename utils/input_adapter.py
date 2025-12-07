# utils/input_adapter.py
from models import PlannerInputs
from utils.xml_loader import DEFAULT_SETUP, DEFAULT_ACCOUNTS

def get_planner_inputs(
    num_simulations: int,
    base_annual_spending: float,
    withdrawal_rate: float,
    portfolio_data: list,
    max_roth: float,
    travel: float,
    gifting: float,
    tax_strategy: str,
    irmaa_strategy: str,
    # any others you use
):
    # Start with defaults from XML
    inputs_dict = DEFAULT_SETUP.copy()
    # Update portfolio
    accounts = {row["name"]: {
        "balance": row.get("balance", 0),
        "equity": row.get("equity", 0.7),
        "bond": row.get("bond", 0.3),
        "tax": row.get("tax", "traditional"),
        "owner": row.get("owner", "person1"),
        "basis": row.get("basis", 0),
        # etc.
    } for row in portfolio_data}

    inputs_dict["accounts"] = accounts

    # ←←← THIS IS THE ONLY PLACE UI VALUES GO ←←←
    inputs_dict["nsims"] = num_simulations
    inputs_dict["base_annual_spending"] = base_annual_spending
    inputs_dict["withdrawal_rate"] = withdrawal_rate
    inputs_dict["max_roth"] = max_roth
    inputs_dict["travel"] = travel
    inputs_dict["gifting"] = gifting
    inputs_dict["tax_strategy"] = tax_strategy
    inputs_dict["irmaa_strategy"] = irmaa_strategy
    inputs_dict["ss_fail_year"] = 2038   # or whatever default
    inputs_dict["ss_fail_percent"] = 0.23

    # Create the object
    return PlannerInputs(**inputs_dict)
