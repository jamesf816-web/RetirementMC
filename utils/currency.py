# utils/currency.py
from dash import dcc, html
from typing import Union

def pretty_currency_input(id, value, label=None, min_val=None, max_val=None, step=None):
    """
    Generates a stylized currency input component.
    """
    
    # --- 1. Label Generation Logic 
    label_text = label

    if label_text is None:
        label_text = " ".join(word.capitalize() for word in id.replace('-', '_').split('_'))
    
    if not label_text and label_text != 0:
        label_text = None
        
    # --- 2. Input Properties 
    input_props = {}
    
    # Use native dcc.Input properties (min, max, step) instead of custom data- attributes
    if min_val is not None:
        input_props['min'] = min_val
    if max_val is not None:
        input_props['max'] = max_val
    if step is not None:
        input_props['step'] = step

    # --- 3. Component Assembly ---
    children = []
    
    if label_text is not None:
        children.append(
            html.Label(
                label_text, 
                style={
                    'fontWeight': 'bold', 
                    'fontSize': 16,
                    'textAlign': 'center',
                    'marginBottom': '6px',
                    'display': 'block'
                }
            )
        )
        
    children.append(
        dcc.Input(
            id=id,
            type='text', 
            value=f"${value:,}",
            placeholder="$2,500,000",
            style={
                'width': '70%',
                'height': '36px',
                'textAlign': 'center',
                'fontSize': '16px',
                'fontFamily': 'monospace',
                'fontWeight': '500',
                'border': '1px solid #ccc',
                'borderRadius': '6px'
            },
            pattern=r'^\$?\s*[0-9,]{0,15}(\.[0-9]{0,2})?$',
            debounce=True,
            **input_props # Now correctly passing 'min', 'max', 'step'
        )
    )
    
    return children


# ----------------------------------------------------------------------
# Helper Function 
# ----------------------------------------------------------------------

def clean_currency(val):
    """
    Cleans a currency string (e.g., "$140,000.00") into a float (140000.0).
    """
    if not val:
        return 0.0
    
    try:
        # Strip non-digit, non-decimal characters, then convert to float.
        cleaned_val = str(val).replace('$', '').replace(',', '').strip()
        if not cleaned_val:
            return 0.0
        return float(cleaned_val)
    except ValueError:
        return 0.0

#----------------------------------------------------------------------
# Helper: Pretty Percent Input
# ----------------------------------------------------------------------

def pretty_percent_input(id, value, label=None, min_val=None, max_val=None, step=None, placeholder="0.00%", decimals=2):
    """
    Generates a stylized percentage input component using type='text' 
    for custom formatting.
    """
    # --- 1. Label Generation Logic (Unchanged) ---
    label_text = label

    if label_text is None:
        label_text = " ".join(word.capitalize() for word in id.replace('-', '_').split('_'))
    
    if not label_text and label_text != 0:
        label_text = None
        
    # --- 2. Format Initial Value for Display ---
    # Convert the underlying float (e.g., 0.23) to the display string (e.g., "23%")
    # We will format it to two decimal places for uniformity.
    initial_display_value = format_percent_output(value, decimals) 
    
    # --- 3. Input Properties (Using native dcc.Input props) ---
    # NOTE: min/max/step will be enforced by the callback, not the dcc.Input type='text'
    # but we keep them here for reference if you switch back or use them in the callback.
    input_props = {}
    if min_val is not None:
        input_props['data-min'] = min_val # Store as data-attribute if type is text
    if max_val is not None:
        input_props['data-max'] = max_val

    # --- 4. Component Assembly ---
    children = []
    
    if label_text is not None:
        children.append(
            html.Label(
                label_text, 
                style={
                    'fontWeight': 'bold', 
                    'fontSize': 16,
                    'textAlign': 'center',
                    'marginBottom': '6px',
                    'display': 'block'
                }
            )
        )
        
    children.append(
        dcc.Input(
            id=id,
            type='text', 
            value=initial_display_value, # Use the formatted string
            placeholder=placeholder,
            style={
                'width': '80%',
                'height': '36px',
                'textAlign': 'center',
                'fontSize': '16px',
                'fontFamily': 'monospace',
                'fontWeight': '500',
                'border': '1px solid #ccc',
                'borderRadius': '6px'
            },
            debounce=True,
            **input_props
        )
    )
    return children

def clean_percent(raw_input: Union[str, float, int]) -> Union[float, None]:
    """
    Cleans raw input (e.g., '0.23', '23%', '23') and converts it to a float
    where 1.0 represents 100%. Handles flexible user input.
    """
    if raw_input is None:
        return None
        
    if isinstance(raw_input, (float, int)):
        # If the input is a number between 1 and 100, treat it as a percentage
        # e.g., 23 -> 0.23
        if 1.0 <= float(raw_input) <= 100.0: 
            return float(raw_input) / 100.0
        # Otherwise, treat it as a decimal, e.g., 0.23 -> 0.23
        return float(raw_input)
        
    s = str(raw_input).strip()
    if not s:
        return None

    # 1. Remove non-numeric characters (%, commas, spaces)
    # Using replace for safety, though only '%' is expected
    s = s.replace('%', '').replace(',', '').replace(' ', '').strip()

    try:
        numeric_val = float(s)
        
        # 2. Apply division if the number is between 1 and 100
        if 1.0 <= numeric_val <= 100.0:
            return numeric_val / 100.0
        
        # 3. Return as is (if it's already a decimal like 0.23)
        return numeric_val
        
    except ValueError:
        return None # Return None or raise an error for unparseable input

def format_percent_output(value: Union[float, None], decimal_places: int = 1) -> str:
    """Formats a float (0.23) to a display string ('23.0%')."""
    if value is None:
        return ""
    value=float(value)
    # Format the number as a percentage string
    return f"{value * 100:.{decimal_places}f}%"

#----------------------------------------------------------------------
# Helper: Pretty Year Input
# ----------------------------------------------------------------------

def pretty_year_input(id, value, label=None, min_val=1900, max_val=2100, step=1, placeholder="2026"):
    """
    Generates a stylized percentage or plain numbeinteger year input component.
    It returns a list of children [Label, Input] using the same style as 
    pretty_currency_input, but with type='number'.
    """
    
    # --- 1. Label Generation Logic (Same as currency) ---
    label_text = label

    if label_text is None:
        label_text = " ".join(word.capitalize() for word in id.replace('-', '_').split('_'))
    
    if not label_text and label_text != 0:
        label_text = None
        
    # --- 2. Input Properties (Using native dcc.Input props) ---
    input_props = {}
    if min_val is not None:
        input_props['min'] = min_val
    if max_val is not None:
        input_props['max'] = max_val
    if step is not None:
        input_props['step'] = step

    # --- 3. Component Assembly ---
    children = []
    
    if label_text is not None:
        children.append(
            html.Label(
                label_text, 
                style={
                    'fontWeight': 'bold', 
                    'fontSize': 16,
                    'textAlign': 'center',
                    'marginBottom': '6px',
                    'display': 'block'
                }
            )
        )
        
    children.append(
        dcc.Input(
            id=id,
            type='number', 
            value=int(value),
            placeholder=placeholder,
            style={
                'width': '80%',
                'height': '36px',
                'textAlign': 'center',
                'fontSize': '16px',
                'fontFamily': 'monospace',
                'fontWeight': '500',
                'border': '1px solid #ccc',
                'borderRadius': '6px'
            },
            debounce=True,
            **input_props
        )
    )
    return children

def format_currency_output(val, decimals=0):
    """
    Formats a float/int into a clean currency string ($1,234,567.00).

    Args:
        val (float): The numerical value to format.
        decimals (int): Number of decimal places.
    """
    if val is None:
        val = 0.0
    return f"${val:,.{decimals}f}"
        
