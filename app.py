# app.py
import dash
from dash import Dash
from layout.main_layout import get_main_layout
from callbacks import editor_callbacks, simulation_callbacks

# Initialize app
app = Dash(__name__, assets_folder="assets", suppress_callback_exceptions=True)
server = app.server

# Assign layout
app.layout = get_main_layout(app)

# Import all callbacks
from callbacks import editor_callbacks
from callbacks import simulation_callbacks
from callbacks import results_callbacks

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8050)

