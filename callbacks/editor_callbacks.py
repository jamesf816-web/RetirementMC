from dash import Input, Output, State, callback, no_update, ctx
from dash.exceptions import PreventUpdate
from utils.xml_loader import parse_portfolio_xml

DEFAULT_PORTFOLIO_XML_PATH = 'config/default_portfolio.xml'

# --- Data Loader Function ---
def get_default_portfolio_data():
    """
    Loads the default account data from the XML file using the utility.
    Includes a fallback in case the XML file is not found.
    """
    try:
        # Use the imported utility function with the defined file path
        return parse_portfolio_xml(DEFAULT_PORTFOLIO_XML_PATH)
    except FileNotFoundError:
        print(f"ERROR: Default portfolio XML file not found at {DEFAULT_PORTFOLIO_XML_PATH}. Returning fallback data.")
        # Fallback to a hardcoded structure (matching the mock data) if the file is missing/inaccessible
        return {
            "Roth_IRA": {"balance": 50000, "equity": 0.80, "bond": 0.20, "tax": "roth", "owner": "person1", "basis": None, "mandatory_yield": None, "rmd_factor_table": None},
            "401k": {"balance": 200000, "equity": 0.70, "bond": 0.30, "tax": "traditional", "owner": "person1", "basis": None, "mandatory_yield": None, "rmd_factor_table": None},
            "Taxable_Brokerage": {"balance": 150000, "equity": 0.90, "bond": 0.10, "tax": "taxable", "owner": "person1", "basis": 150000, "mandatory_yield": None, "rmd_factor_table": None},
        }

    
def register_editor_callbacks(app):

    # ----------------------------------------------------------------------
    # Portfolio Grid / Store Callbacks
    # ----------------------------------------------------------------------

    @app.callback(
        Output('portfolio-store', 'data'),
        Input('portfolio-grid', 'cellValueChanged'),
        State('portfolio-grid', 'rowData'),
        State('portfolio-store', 'data'),
        prevent_initial_call=True
    )

    def update_portfolio_store(changed_cell, row_data, current_store):
        if not ctx.triggered:
            return no_update

        # Rebuild dict from current grid rows (name → row dict)
        new_data = {}
        for row in row_data:
            name = row.get("name")
            if name:
                new_data[name] = row

        return new_data

    # Optional: Reset to defaults button
    @app.callback(
        Output("portfolio-grid", "rowData"),
        Input("reset-portfolio-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def reset_portfolio(n_clicks):
        if n_clicks:
            DEFAULT_ACCOUNTS = get_default_portfolio_data() 
            return [{**v, "name": k} for k, v in DEFAULT_ACCOUNTS.items()]
        return no_update

    @app.callback(
        Output("setup-store", "data"),
        Input("setup-grid", "rowData"),
    )
    def update_setup_store(rows):
        return rows

    # Add new account
    @app.callback(
        Output('portfolio-grid', 'rowData', allow_duplicate=True),
        Input('add-account-btn', 'n_clicks'),
        State('portfolio-grid', 'rowData'),
        prevent_initial_call=True
    )
    def add_account(n, current_rows):
        if n == 0:
            raise PreventUpdate
        new_name = f"New_Account_{len(current_rows)+1}"
        new_row = {
            "name": new_name,
            "balance": 100000,
            "equity": 0.70,
            "bond": 0.30,
            "tax": "traditional",
            "owner": "person1",
            "basis": None,
            "mandatory_yield": None,
            "rmd_factor_table": None
        }
        return current_rows + [new_row]


    # Toggle the portfolio editor open/closed
    @app.callback(
        Output("portfolio-collapse-content", "style"),
        Output("portfolio-collapse-button", "children"),
        Input("portfolio-collapse-button", "n_clicks"),
        State("portfolio-collapse-content", "style"),
        prevent_initial_call=True
    )
    def toggle_portfolio_collapse(n_clicks, current_style):
        if not n_clicks:
            raise PreventUpdate
        if current_style and current_style.get("display") == "block":
            return {"display": "none"}, "Portfolio Editor – Click to Open"
        else:
            return {"display": "block"}, "Portfolio Editor – Click to Close"


    # Toggle the setup editor open/closed
    @app.callback(
        Output("setup-collapse-content", "style"),
        Output("setup-collapse-button", "children"),
        Input("setup-collapse-button", "n_clicks"),
        State("setup-collapse-content", "style"),
        prevent_initial_call=True
    )
    def toggle_setup_collapse(n_clicks, current_style):
        if not n_clicks:
            raise PreventUpdate
        if current_style and current_style.get("display") == "block":
            return {"display": "none"}, "Setup Editor – Click to Open"
        else:
            return {"display": "block"}, "Setup Editor – Click to Close"

    # ----------------------------------------------------------------------
    # UI Transition Callback: Hides setup, reveals main dashboard
    # ----------------------------------------------------------------------
    @app.callback(
        # Output 1: Hide the initial setup UI
        Output('initial-setup-container', 'style'),
        # Output 2: Show the main planning UI
        Output('main-planning-ui', 'style'),
        # Input: Listen to the "Start Planning" button
        Input('confirm-setup-btn', 'n_clicks'), 
        prevent_initial_call=True
    )
    def handle_setup_confirmation(n_clicks):
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        
        # When clicked, hide the setup container and display the main UI
        hide_style = {'display': 'none'}
        show_style = {'display': 'block'} 
        
        return hide_style, show_style

