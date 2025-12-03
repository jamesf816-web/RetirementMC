from dash import Input, Output, State, callback
from dash.exceptions import PreventUpdate
from config.default_portfolio import accounts as DEFAULT_ACCOUNTS

# ----------------------------------------------------------------------
# Portfolio Grid / Store Callbacks
# ----------------------------------------------------------------------

@callback(
    Output('portfolio-store', 'data'),
    Input('portfolio-grid', 'cellValueChanged'),
    State('portfolio-grid', 'rowData'),
    prevent_initial_call=True
)
def update_portfolio_store(event, grid_data):
    if not grid_data:
        return DEFAULT_ACCOUNTS

    new_accounts = {}
    for row in grid_data:
        name = row["name"]
        # Clean up None values
        clean_row = {
            k: (v if v not in [None, ""] else None)
            for k, v in row.items()
            if k != "name"
        }
        keep_if_none = ["basis", "mandatory_yield", "rmd_factor_table"]
        clean_row = {
            k: v
            for k, v in clean_row.items()
            if v is not None or k in keep_if_none
        }
        new_accounts[name] = clean_row
    return new_accounts


@callback(
    Output("setup-store", "data"),
    Input("setup-grid", "rowData"),
)
def update_setup_store(rows):
    return rows


@callback(
    Output("portfolio-status", "children"),
    Input('portfolio-store', 'data'),
)
def update_portfolio_total(store_data):
    if not store_data:
        return "Total Portfolio: $0"
    total = sum((acct.get("balance") or 0) for acct in store_data.values())
    return f"Total Portfolio: ${total:,.0f}  •  {len(store_data)} accounts"


# Add new account
@callback(
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


# Reset to defaults
@callback(
    Output('portfolio-grid', 'rowData', allow_duplicate=True),
    Output('portfolio-store', 'data', allow_duplicate=True),
    Input('reset-portfolio-btn', 'n_clicks'),
    prevent_initial_call=True
)
def reset_portfolio(n):
    if n == 0:
        raise PreventUpdate
    return [{**v, "name": k} for k, v in DEFAULT_ACCOUNTS.items()], DEFAULT_ACCOUNTS


# Toggle the portfolio editor open/closed
@callback(
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
@callback(
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




