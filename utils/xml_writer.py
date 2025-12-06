import xml.etree.ElementTree as ET
from xml.dom import minidom
import json

def prettify_xml(elem):
    """Return a pretty-printed XML string for an Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    # Return the XML declaration and the pretty-printed content
    return reparsed.toprettyxml(indent="    ")

def create_portfolio_xml(portfolio_dict: dict) -> str:
    """
    Converts the in-memory portfolio dictionary (e.g., from dcc.Store) 
    back into a properly formatted XML string.
    
    The portfolio_dict is expected to be a dictionary where keys are 
    account names and values are dictionaries of account properties.
    """
    
    # Create the root element
    root = ET.Element('portfolio')
    
    for account_name, attributes in portfolio_dict.items():
        # Create the <account> tag
        account_elem = ET.SubElement(root, 'account', name=account_name)
        
        # Iterate over all attributes (balance, equity, tax, etc.)
        for key, value in attributes.items():
            # Skip the account 'name' which is handled as an attribute
            if key in ('name', 'id'):
                continue

            # Ensure the value is not None and is convertible to string
            if value is not None:
                # Create the child tag and set its text content
                # Use str() to convert numbers/floats to strings for XML
                ET.SubElement(account_elem, key).text = str(value)

    # Convert the root element tree to a prettified XML string
    return prettify_xml(root)

# Example usage (for testing)
if __name__ == '__main__':
    # This structure mirrors the data in your dcc.Store
    test_data = json.loads(
        '{"GST_Exempt": {"balance": 3132760, "equity": 0.75, "bond": 0.25, "tax": "trust", "owner": "person1", "mandatory_yield": 0.015}, '
         '"Taxable": {"balance": 1347857, "equity": 0.92, "bond": 0.08, "tax": "taxable", "owner": "person1", "basis": 789874}}'
    )
    xml_output = create_portfolio_xml(test_data)
    print(xml_output)
