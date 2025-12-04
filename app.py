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
from layout.main_layout import main_layout
from models import PlannerInputs
from utils.input_adapter import get_planner_inputs

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

from callbacks.simulation_callbacks import register_simulation_callbacks
register_simulation_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8050)

