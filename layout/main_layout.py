# layout/main_layout.py
import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Assuming these are imported from your other modules (adjust paths as needed)
from config.default_setup import default_setup
from config.default_portfolio import accounts
from utils.plotting import create_balance_paths_fig, create_ending_balance_hist  # Adjust if needed
from engine.monte_carlo import MonteCarloSimulator  # For sim runs

dash.register_page(__name__, path="/")  # If using multi-page, optional

def get_main_layout():
    return html.Div(
        className="app-container",
        style={
            "display": "flex",
            "height": "100vh",
            "width": "100%",
            "fontFamily": "Arial, sans-serif",
        },
        children=[
            # Sidebar: Inputs
            html.Div(
                className="sidebar",
                style={
                    "width": "300px",
                    "backgroundColor": "#f8f9fa",
                    "padding": "20px",
                    "overflowY": "auto",
                    "borderRight": "1px solid #dee2e6",
                },
                children=[
                    html.H2("Retirement Planner", style={"textAlign": "center", "color": "#495057"}),
                    
                    # Portfolio Editor Section (Collapsible, button left of slider)
                    html.Div(
                        style={"width": "100%", "marginBottom": "20px"},
                        children=[
                            html.Div(
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "width": "100%",
                                    "flexDirection": "row",
                                },
                                children=[
                                    html.Button(
                                        "Edit Portfolio",
                                        id="portfolio-editor-toggle",
                                        n_clicks=0,
                                        style={
                                            "flex": "0 0 auto",
                                            "marginRight": "10px",
                                            "padding": "8px 12px",
                                            "backgroundColor": "#007bff",
                                            "color": "white",
                                            "border": "none",
                                            "borderRadius": "4px",
                                            "cursor": "pointer",
                                        },
                                    ),
                                    dcc.Slider(
                                        id="allocation-slider",
                                        min=0,
                                        max=100,
                                        step=5,
                                        value=60,  # Default stock allocation
                                        marks={i: f"{i}%" for i in range(0, 101, 10)},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                        style={"flex": 1, "margin": "0"},
                                    ),
                                ],
                            ),
                            # Collapsible Portfolio Editor (show/hide div)
                            html.Div(
                                id="portfolio-editor-content",
                                style={
                                    "display": "none",  # Hidden by default
                                    "width": "100%",
                                    "marginTop": "10px",
                                    "backgroundColor": "white",
                                    "border": "1px solid #dee2e6",
                                    "borderRadius": "4px",
                                    "padding": "10px",
                                },
                                children=[
                                    dag.AgGrid(
                                        id="portfolio-grid",
                                        columnDefs=[
                                            {"field": "asset", "headerName": "Asset", "editable": True},
                                            {"field": "allocation", "headerName": "Allocation %", "editable": True},
                                            {
                                                "field": "delete",
                                                "headerName": "",
                                                "cellRenderer": "agRichTextCellRenderer",
                                                "cellRendererParams": {"value": "🗑️ Delete"},
                                                "onCellClicked": "deleteRow",
                                            },
                                        ],
                                        rowData=[
                                            {"asset": "Stocks", "allocation": 60},
                                            {"asset": "Bonds", "allocation": 40},
                                        ],
                                        defaultColDef={"flex": 1, "resizable": True},
                                        style={"height": "200px", "width": "100%"},
                                    ),
                                    html.Button(
                                        "Add Row",
                                        id="add-portfolio-row",
                                        n_clicks=0,
                                        style={
                                            "marginTop": "10px",
                                            "padding": "5px 10px",
                                            "backgroundColor": "#28a745",
                                            "color": "white",
                                            "border": "none",
                                            "borderRadius": "4px",
                                        },
                                    ),
                                ],
                            ),
                        ],
                    ),
                    
                    # Other Inputs (Sliders, Fields, Grids for Expenses, etc.)
                    html.Div(
                        style={"width": "100%", "marginBottom": "20px"},
                        children=[
                            html.H3("Personal Inputs", style={"marginBottom": "10px"}),
                            dcc.Input(id="current-age", type="number", value=45, placeholder="Current Age"),
                            dcc.Input(id="retirement-age", type="number", value=65, placeholder="Retirement Age"),
                            # Add more as per original: savings, expenses, etc.
                            # Example grid for expenses (with delete)
                            dag.AgGrid(
                                id="expenses-grid",
                                columnDefs=[
                                    {"field": "category", "headerName": "Category", "editable": True},
                                    {"field": "amount", "headerName": "Annual Amount", "editable": True},
                                    {"field": "delete", "headerName": "", "cellRenderer": lambda: "🗑️"},
                                ],
                                rowData=[{"category": "Housing", "amount": 24000}],
                                defaultColDef={"flex": 1},
                                style={"height": "150px", "width": "100%"},
                            ),
                        ],
                    ),
                    
                    # Run Button
                    html.Button(
                        "Run Simulation",
                        id="run-simulation-button",
                        n_clicks=0,
                        style={
                            "width": "100%",
                            "padding": "12px",
                            "backgroundColor": "#28a745",
                            "color": "white",
                            "border": "none",
                            "borderRadius": "4px",
                            "fontSize": "16px",
                            "cursor": "pointer",
                            "marginTop": "20px",
                        },
                    ),
                ],
            ),
            
            # Main Content: Results
            html.Div(
                className="main-content",
                style={
                    "flex": 1,
                    "padding": "20px",
                    "overflowY": "auto",
                    "backgroundColor": "white",
                },
                children=[
                    html.H1("Simulation Results", style={"textAlign": "center", "color": "#495057"}),
                    html.Div(id="results-output", children=[]),  # Dynamic content here
                    dcc.Graph(id="balance-paths-graph", style={"height": "400px"}),
                    dcc.Graph(id="ending-balance-hist", style={"height": "400px"}),
                    html.Div(id="metrics-table"),  # For success rate, percentiles, etc.
                ],
            ),
        ],
    )

# Example Callbacks (move to callbacks/ if separate; these restore core functionality)
# Toggle Portfolio Editor
@callback(
    Output("portfolio-editor-content", "style"),
    Input("portfolio-editor-toggle", "n_clicks"),
    State("portfolio-editor-content", "style"),
)
def toggle_portfolio_editor(n_clicks, current_style):
    if n_clicks % 2 == 1:
        return {**current_style, "display": "block"}
    return {**current_style, "display": "none"}

# Run Simulation (integrates with engine)
@callback(
    [Output("balance-paths-graph", "figure"),
     Output("ending-balance-hist", "figure"),
     Output("metrics-table", "children")],
    Input("run-simulation-button", "n_clicks"),
    [State(id, "value") for id in ["current-age", "retirement-age", "allocation-slider"]] +  # Add all states
    [State("portfolio-grid", "rowData"), State("expenses-grid", "rowData")],
    prevent_initial_call=True,
)
def update_results(n_clicks, age, ret_age, allocation, portfolio_data, expenses_data):
    if not n_clicks:
        return go.Figure(), go.Figure(), html.P("No results yet.")
    
    # Gather inputs (use your get_default_inputs or similar)
    inputs = get_default_inputs()
    inputs.current_age = age
    inputs.retirement_age = ret_age
    inputs.stock_allocation = allocation
    # Update from grids: e.g., inputs.portfolio = portfolio_data
    
    # Run sim
    simulator = MonteCarloSimulator(inputs)
    results = simulator.run(num_trials=10000)
    
    # Generate plots/table
    fig_paths = create_balance_paths_fig(results.balances)
    fig_hist = create_ending_balance_hist(results.ending_balances)
    metrics = html.Table(
        [
            html.Tr([html.Th("Metric"), html.Th("Value")]),
            html.Tr([html.Td("Success Rate"), html.Td(f"{results.success_rate:.1%}")]),
            html.Tr([html.Td("Median Ending Balance"), html.Td(f"${results.median_balance:,.0f}")]),
            # Add more: 10th/90th percentiles, SWR, etc.
        ],
        style={"width": "100%", "borderCollapse": "collapse"},
    )
    
    return fig_paths, fig_hist, metrics

# Grid Delete (clientside example; attach to ag-grid)
# In main.py, add: app.clientside_callback("function deleteRow(params) { ... }", ...)

if __name__ == "__main__":
    # For testing: app = dash.Dash(__name__); app.layout = get_main_layout(); app.run_server()
    pass
