# utils/input_adapter.py
from typing import Dict
from dataclasses import fields
from utils.xml_loader import parse_setup_xml, parse_portfolio_xml
from models import PlannerInputs

def get_planner_inputs(**overrides):
    # Load defaults from XML
    default_setup = parse_setup_xml("config/default_setup.xml")
    default_portfolio = parse_portfolio_xml("config/default_portfolio.xml")

    # Start with defaults
    inputs_dict = {}
    inputs_dict.update(default_setup)
    inputs_dict["accounts"] = default_portfolio

    # Apply UI overrides
    if "portfolio_data" in overrides:
        # Merge override rows with XML defaults
        for row in overrides["portfolio_data"]:
            name = row.get("name")
            if name:
                inputs_dict["accounts"][name] = {**inputs_dict["accounts"].get(name, {}), **row}
        del overrides["portfolio_data"]

    inputs_dict.update(overrides)

    # Hard-coded defaults for UI-driven or simulation-level parameters
    ui_defaults = {
        "nsims": 1000,
        "tax_strategy": "fill_24_percent",
        "irmaa_strategy": "fill_IRMAA_3", 
        "max_roth": 240000.0,            
        "travel": 50000.0,               
        "gifting": 48000.0,              
        "ss_fail_year": 2033,            
        "ss_fail_percent": 0.23,
    }

    # Merge hard-coded defaults into inputs dictionary, without overwriting XML/UI overrides
    for k, v in ui_defaults.items():
        if k not in inputs_dict:
            inputs_dict[k] = v


    # Filter to only the fields that exist in PlannerInputs
    planner_fields = {f.name for f in fields(PlannerInputs)}
    filtered_inputs = {k: v for k, v in inputs_dict.items() if k in planner_fields}

    # Convert to dataclass
    planner_inputs = PlannerInputs(**filtered_inputs)
    return planner_inputs

