# utils/ui_components.py
import numpy as np
from dash import html

def vanguard_color(success_rate):
    """Returns a color based on the success rate for styling (e.g., Green for high, Red for low)."""
    
    # Check for valid numeric input. If not, return a safe, default color.
    if not isinstance(success_rate, (int, float)) or success_rate is None or np.isnan(success_rate):
        return 'rgb(128, 128, 128)'  # Gray color for safety

    sr = max(0, min(100, success_rate))
    
    # Example color logic (adjust based on your original logic)
    if sr >= 80:
        return 'rgb(0, 128, 0)' # Green
    elif sr >= 50:
        return 'rgb(255, 165, 0)' # Orange
    else:
        return 'rgb(255, 0, 0)' # Red

def create_rate_header(success_rate, avoid_ruin_rate):
    """Creates the HTML Div for the simulation success/ruin rate header."""
    
    color = vanguard_color(success_rate)
    success_style = {
        "color": color,
        "fontWeight": "bold",
        "fontSize": "22px",
        "margin": "0 10px",
        "textAlign": "center"
    }
    
    # Create the final Div
    return html.Div([
        html.H3(f"Success Rate: {success_rate:.1f}%", style=success_style),
        html.H3(f"Ruin Avoidance: {avoid_ruin_rate:.1f}%", style=success_style),
    ], style={'display': 'flex', 'justifyContent': 'center', 'padding': '15px'})
