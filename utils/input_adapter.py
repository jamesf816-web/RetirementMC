from models import PlannerInputs
from utils.xml_loader import DEFAULT_SETUP
from dataclasses import fields
from typing import List, Dict, Any

def get_planner_inputs(
    portfolio_data: List[Dict], # Only list arguments that need special processing
    **kwargs: Any              # Catch all other UI inputs dynamically
) -> PlannerInputs:
    """
    Dynamically generates PlannerInputs by merging XML defaults and all UI inputs,
    using reflection (dataclasses.fields) to ensure only valid fields are passed.
    """
    
    # 1. Start with defaults loaded from the XML setup file
    inputs_dict = DEFAULT_SETUP.copy()
    
    # 2. Merge ALL UI inputs (passed via **kwargs) into the defaults.
    # The UI inputs will override any matching fields in DEFAULT_SETUP.
    inputs_dict.update(kwargs)

    # 3. Handle the 'nsims' naming convention mismatch if necessary.
    # Assuming the UI input name is 'num_simulations' and the dataclass field is 'nsims'.
    if 'num_simulations' in inputs_dict:
        inputs_dict['nsims'] = inputs_dict.pop('num_simulations')

    # 4. Handle the special case: portfolio_data -> accounts dictionary
    accounts = {}
    for row in portfolio_data:
        acct_name = row.get("name")
        if acct_name:
            # Note: Ensure all account-specific fields are included here
            accounts[acct_name] = {
                "balance": row.get("balance", 0.0),
                "equity": row.get("equity", 0.7),
                "bond": row.get("bond", 0.3),
                "tax": row.get("tax", "traditional"),
                "owner": row.get("owner", "person1"),
                "basis": row.get("basis", 0.0),
                "rmd_factor_table": row.get("rmd_factor_table"),
            }
    inputs_dict["accounts"] = accounts

    # 5. DYNAMIC FIELD MAPPING AND FILTERING (Reflection)
    # Get the set of all valid field names defined in the PlannerInputs dataclass.
    planner_field_names = {f.name for f in fields(PlannerInputs)}
    
    # Filter the inputs_dict to ensure only keys that match PlannerInputs fields are kept.
    final_inputs = {
        key: value 
        for key, value in inputs_dict.items() 
        if key in planner_field_names
    }
    
    # 6. Create the PlannerInputs object
    return PlannerInputs(**final_inputs)
