# utils/xml_loader.py
import xml.etree.ElementTree as ET
from typing import Any, Dict
from pathlib import Path

def parse_setup_xml(file_path: str) -> Dict[str, Any]:
    tree = ET.parse(file_path)
    root = tree.getroot()

    setup_dict: Dict[str, Any] = {}

    # Top-level elements like current_year, end_age
    for child in root:
        if child.tag in ["person1", "person2", "simulation"]:
            # Nested elements
            for sub in child:
                setup_dict[f"{child.tag}_{sub.tag}"] = try_cast(sub.text)
        else:
            setup_dict[child.tag] = try_cast(child.text)
    
    return setup_dict

def try_cast(value: str) -> Any:
    """Try to convert string to int or float if possible, else leave as str."""
    if value is None:
        return None
    value = value.strip()
    # Booleans
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    # Integers
    try:
        return int(value)
    except ValueError:
        pass
    # Floats
    try:
        return float(value)
    except ValueError:
        pass
    # Leave as string
    return value

def parse_portfolio_xml(file_path: str) -> Dict[str, Dict]:
    """Load portfolio accounts from XML into a dict of dicts."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    portfolio_dict: Dict[str, Dict] = {}

    for acct in root.findall("account"):
        name = acct.get("name", f"Account_{len(portfolio_dict)+1}")
        acct_dict = {}
        for field in acct:
            acct_dict[field.tag] = try_cast(field.text)
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
