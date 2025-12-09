# utils/xml_loader.py
import xml.etree.ElementTree as ET
from typing import Any, Dict
from pathlib import Path

def parse_setup_xml(file_path: str) -> Dict[str, Any]:
    tree = ET.parse(file_path)
    root = tree.getroot()

    setup_dict: Dict[str, Any] = {}

    for child in root:
        if child.tag in ["person1", "person2", "simulation"]:
            for sub in child:
                val = try_cast(sub.text)
                # Normalize owner references and booleans
                if sub.tag in ["name", "owner"] and isinstance(val, str):
                    val = val.strip().lower()
                setup_dict[f"{child.tag}_{sub.tag}"] = val
        else:
            val = try_cast(child.text)
            if child.tag in ["current_year", "end_age", "start_age", "death_year", "death_month"]:
                val = int(val) if val is not None else val
            if child.tag in ["balance", "equity", "bond", "basis"]:
                val = float(val) if val is not None else val
            setup_dict[child.tag] = val

    return setup_dict

def try_cast(value: str) -> Any:
    """Try to convert string to int or float if possible, else leave as str."""
    if value is None:
        return None
    value = value.strip()
    # Booleans
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    
    # Integers (try first)
    try:
        if '.' not in value: # Optimization: check for decimal to avoid unnecessary exception
            return int(value)
    except ValueError:
        pass
        
    # Floats (try second)
    try:
        return float(value)
    except ValueError:
        pass
        
    return value # Return as string if all else fails

def parse_portfolio_xml(file_path: str) -> Dict[str, Dict]:
    """Load portfolio accounts from XML into a dict of dicts, with normalized values."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    portfolio_dict: Dict[str, Dict] = {}

    for acct in root.findall("account"):
        name = acct.get("name", f"Account_{len(portfolio_dict)+1}")
        acct_dict = {}
        for field in acct:
            value = try_cast(field.text)

            # Normalize key fields
            if field.tag == "tax" and isinstance(value, str):
                value = value.strip().lower()
            if field.tag == "owner" and isinstance(value, str):
                value = value.strip().lower()
            if field.tag in ["equity", "bond", "balance", "basis", "income"] and value is not None:
                value = float(value)  # ensure numeric types are floats

            acct_dict[field.tag] = value

        portfolio_dict[name] = acct_dict

    return portfolio_dict


def parse_xml_content_to_dict(file_like_object: Any) -> Dict[str, Dict]:
    """
    Load portfolio accounts from XML content (provided as a file-like object) 
    into a dict of dicts.
    
    This function is designed to work with the content decoded from dcc.Upload.
    """
    # ET.parse can accept a file name (string) or a file-like object
    tree = ET.parse(file_like_object)
    root = tree.getroot()
    portfolio_dict: Dict[str, Dict] = {}

    for acct in root.findall("account"):
        # Use existing logic to extract account name and details
        name = acct.get("name", f"Account_{len(portfolio_dict)+1}")
        acct_dict = {}
        for field in acct:
            acct_dict[field.tag] = try_cast(field.text)
        portfolio_dict[name] = acct_dict

    return portfolio_dict

CONFIG_DIR = Path(__file__).parent.parent / "config"

DEFAULT_SETUP = parse_setup_xml(CONFIG_DIR / "default_setup.xml")
DEFAULT_ACCOUNTS = parse_portfolio_xml(CONFIG_DIR / "default_portfolio.xml")
