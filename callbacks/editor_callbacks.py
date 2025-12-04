from dash import Input, Output, State, callback
from dash.exceptions import PreventUpdate
from config.default_portfolio import accounts as DEFAULT_ACCOUNTS

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
            from config.default_portfolio import accounts as DEFAULT_ACCOUNTS
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

