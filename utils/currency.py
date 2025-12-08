# utils/currency.py
from dash import dcc, html

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
                'width': '100%',
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

def pretty_percent_input(id, value, label=None, min_val=None, max_val=None, step=None, placeholder="0.00%"):
    """
    Generates a stylized percentage or plain number input component.
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
            type='number', # Use 'number' type for accurate decimal/step control
            value=value,
            placeholder=placeholder,
            style={
                'width': '100%',
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
    
    # Returns a list of components [Label, Input]
    return children
