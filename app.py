# app.py
import multiprocessing as mp
import numpy as np
import dash
from dash import Dash
# If you are using Dash 2.0+, these imports are cleaner:
from dash import dcc
from dash import html
import dash_ag_grid as dag 

# -----------------------------------------------------------
# Core Imports
# -----------------------------------------------------------

from layout.main_layout import main_layout
from utils.input_adapter import get_planner_inputs

# Import callback modules
from callbacks.editor_callbacks import register_editor_callbacks      
from callbacks.simulation_callbacks import register_simulation_callbacks

# Initialize app
app = Dash(__name__, assets_folder="assets", suppress_callback_exceptions=True)
server = app.server

# -----------------------------------------------------------
# Layout Assignment & Callback Registration
# -----------------------------------------------------------

# Assign layout (using the imported variable)
app.layout = main_layout

register_simulation_callbacks(app)
register_editor_callbacks(app)
# don't load this here.  
# register_results_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8050)

