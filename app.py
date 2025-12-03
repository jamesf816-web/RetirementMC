import dash
from dash import Dash
# If you are using Dash 2.0+, these imports are cleaner:
from dash import dcc
from dash import html
import dash_ag_grid as dag 

# -----------------------------------------------------------
# Core Imports
# Assuming DEFAULT_ACCOUNTS and main_layout are exported as variables
# -----------------------------------------------------------
from config.default_portfolio import accounts as DEFAULT_ACCOUNTS
from layout.main_layout import main_layout # <-- FIX 1: Import the variable 'main_layout'

# Import callback modules
import callbacks.editor_callbacks as editor_callbacks
import callbacks.simulation_callbacks as simulation_callbacks
import callbacks.results_callbacks as results_callbacks


# Initialize app
app = Dash(__name__, assets_folder="assets", suppress_callback_exceptions=True)
server = app.server

# -----------------------------------------------------------
# Layout Assignment & Callback Registration
# -----------------------------------------------------------

# Assign layout (using the imported variable)
app.layout = main_layout

# FIX 2: Explicitly call the registration function in each callback module
# You must ensure each of these modules defines a function called register_callbacks(app)
editor_callbacks.register_callbacks(app)
simulation_callbacks.register_callbacks(app)
results_callbacks.register_callbacks(app)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8050)

