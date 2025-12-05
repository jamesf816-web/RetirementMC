# main_layout.py
import dash
from dash import Dash
from dash import dcc
from dash import html

import dash_ag_grid as dag
import plotly.graph_objects as go

from utils.xml_loader import DEFAULT_SETUP, DEFAULT_ACCOUNTS
from utils.currency import pretty_currency_input
from utils.currency import pretty_percent_input

from layout.results_layout import create_results_layout

# ----------------------------------------------------------------------
# Application Layout Definition
# ----------------------------------------------------------------------

main_layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'margin': '2%', 'backgroundColor': '#f9f9fb'},
    children=[
        html.H1(
            "Retirement Monte Carlo Explorer",
            style={'textAlign': 'center', 'color': 'black', 'marginBottom': 0}
        ),
        
        # add stores for default portfolio inputs and setup inputs
        dcc.Store(id='portfolio-store', data=DEFAULT_ACCOUNTS),
        dcc.Store(id='setup-store', data=DEFAULT_SETUP),

        # add store for simulation outputs
        dcc.Store(id="simulation-data-store"),

        # ----------------------------------------------------------------------
        # ROW 1: Simulation Controls - Portfolio Editor, Slider and Run Button
        # ----------------------------------------------------------------------

        html.Div([
        # COLUMN 1A: COLLAPSIBLE PORTFOLIO EDITOR
            html.Button(
                "Portfolio Editor – Click to Open/Close",
                id="portfolio-collapse-button",
                n_clicks=0,
                style={
                    'padding': '12px 20px',
                    'fontSize': '16px',
                    'fontWeight': 'bold',
                    'backgroundColor': 'purple',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '8px',
                    'cursor': 'pointer',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                    'whiteSpace': 'nowrap',
                    'height': '50px',
                    'alignSelf': 'flex-end',
                    'marginRight': '20px'
                }
            ),

            # COLUMN 1B: Slider (takes up most of the space)
            html.Div([
                html.Label("Number of Simulations", style={'fontSize': 16, 'fontWeight': 'bold'}),
                dcc.Slider(
                    id='nsims', min=100, max=30000, step=1000, value=1000,
                    marks={i: f"{i//1000}k" for i in range(0, 31000, 5000)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], style={'flex': 1, 'minWidth': '300px', 'padding': '0 20px'}), 

            # COLUMN 1C: Run Button 
            html.Div([
                html.Button(
                    "Run Simulation",
                    id="run",
                    n_clicks=0,
                    style={
                        'marginTop': '30px',
                        'width': '100%',
                        'height': '50px',
                        'fontSize': 16, 
                        'backgroundColor': '#3498db',
                        'color': 'white', 
                        'border': 'none',
                        'borderRadius': '8px',
                        'alignSelf': 'flex-end',
                        'marginBottom': '2px'
                    }
                ),
            ], style={'width': '160px', 'flex': 'none'})

        ], style={
            'display': 'flex', 
            'alignItems': 'flex-end', 
            'gap': '15px', 
            'marginBottom': '20px', 
            'flexWrap': 'wrap', 
            'width': '100%',
            'boxSizing': 'border-box'
        }),

        # COLLAPSIBLE PORTFOLIO EDITOR FULL WIDTH WHEN OPEN
        # This Div shows/hides based on CSS (starts hidden)
        html.Div(
            id="portfolio-collapse-content",
            style={'display': 'none'}, 
            children=[
                html.Div([
                    html.Div([
                        html.Button("Add New Account", id="add-account-btn", n_clicks=0,
                                     style={'marginRight': '10px', 'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'borderRadius': '4px'}),
                        html.Button("Reset to Defaults", id="reset-portfolio-btn", n_clicks=0,
                                     style={'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'borderRadius': '4px'}),
                        html.Div(id="portfolio-status", style={'display': 'inline-block', 'marginLeft': '20px', 'color': '#7f8c8d', 'fontSize': '14px'})
                    ], style={'marginBottom': '15px', 'display': 'flex', 'alignItems': 'center'}),

                    dag.AgGrid(
                        id='portfolio-grid',
                        columnDefs=[
                            {"field": "name", "headerName": "Account Name", "editable": False, "pinned": "left", "width": 180},
                            {"field": "balance", "headerName": "Balance ($)", "type": "rightAligned",
                             "valueFormatter": {"function": "params.value == null ? '' : '$' + Number(params.value).toLocaleString()"},
                             "editable": True},
                            {"field": "equity", "headerName": "Equity %",
                             "valueFormatter": {"function": "params.value == null ? '' : (params.value*100).toFixed(1) + '%'"},
                             "editable": True},
                            {"field": "bond", "headerName": "Bond %",
                             "valueFormatter": {"function": "params.value == null ? '' : (params.value*100).toFixed(1) + '%'"},
                             "editable": True},
                            {"field": "tax", "headerName": "Tax Type", "cellEditor": "agSelectCellEditor",
                             "cellEditorParams": {"values": ["taxable", "traditional", "roth", "inherited", "trust"]}, "width": 130},
                            {"field": "owner", "headerName": "Owner", "cellEditor": "agSelectCellEditor",
                             "cellEditorParams": {"values": ["person1", "person2"]}, "width": 110},
                            {"field": "basis", "headerName": "Basis ($)",
                             "valueFormatter": {"function": "params.value == null ? '' : '$' + Number(params.value).toLocaleString()"},
                             "editable": True},
                            {"field": "mandatory_yield", "headerName": "Mand. Yield", "editable": True, "width": 110},
                            {"field": "rmd_factor_table", "headerName": "RMD Table", "editable": True, "width": 130},
                     {
                        "Headername": "Delete",
                        "field": "delete",
                        "checkboxSelection": True,
                        "width": 90,
                        "pinned": "right",
                        "sortable": False,
                        "filter": False,
                        "headerCheckboxSelection": True,
                        "headerCheckboxSelectionFilteredOnly": True,
                    },
                ],
                        rowData=[{**v, "name": k} for k, v in DEFAULT_ACCOUNTS.items()],
                        defaultColDef={
                            "flex": 1,
                            "minWidth": 100,
                            "resizable": True,
                            "sortable": True,
                            "filter": True,
                            "floatingFilter": True
                        },
                        dashGridOptions={"rowHeight": 48, "animateRows": False},
                        style={"height": 550},
                        className="ag-theme-alpine",
                    )
                ], style={
                    'padding': '25px',
                    'border': '2px solid #ddd',
                    'borderRadius': '12px',
                    'backgroundColor': '#fff',
                    'boxShadow': '0 8px 25px rgba(0,0,0,0.1)',
                    'marginBottom': '30px'
                })
            ]
        ),

        # ----------------------------------------------------------------------
        # ROW 2: Inputs and Dropdowns (7 Items) – YOUR ORIGINAL PERFECT LAYOUT
        # ----------------------------------------------------------------------
        html.Div([
            # Item 2A: Base Spending - Correct call: pretty_currency_input returns a list of children
            html.Div(
                pretty_currency_input('base_annual_spending', value=140000, label="Base Spending"),
                style={'flex': '1', 'minWidth': '150px', 'textAlign': 'center'}
            ),

            # Item 2B: Withdrawal Rate - Now using the new pretty_percent_input for matching style
            html.Div(
                pretty_percent_input('withdrawal_rate', value=0.050, step=0.001, label="Withdrawal Rate", placeholder="5.0%"),
                style={'flex': '1', 'minWidth': '150px', 'textAlign': 'center'}
            ),

            # Item 2C: Max Roth Conv
            html.Div(
                pretty_currency_input('max_roth', value=240000, label="Max Roth Conv"),
                style={'flex': '1', 'minWidth': '150px', 'textAlign': 'center'}
            ),

            # Item 2D: Target Travel
            html.Div(
                pretty_currency_input('travel', value=50000, label="Target Travel"),
                style={'flex': '1', 'minWidth': '140px', 'textAlign': 'center'}
            ),

            # Item 2E: Target Gifting
            html.Div(
                pretty_currency_input('gifting', value=42000, label="Target Gifting"),
                style={'flex': '1', 'minWidth': '140px', 'textAlign': 'center'}
            ),

            # Item 2F: Tax Strategy (Dropdown) - Added display: block to Label for consistency
            html.Div([
                html.Label("Tax Strategy", style={'fontWeight': 'bold', 'display': 'block', 'fontSize': 16, 'marginBottom': '6px'}),
                dcc.Dropdown(
                    id='tax_strategy',
                    options=[
                        {'label': 'Fill 22% bracket ($211k)', 'value': 'fill_22_percent'},
                        {'label': 'Fill 24% bracket ($404k)', 'value': 'fill_24_percent'},
                        {'label': 'Fill 32% bracket ($512k)', 'value': 'fill_32_percent'},
                        {'label': 'Fill 35% bracket ($767k)', 'value': 'fill_35_percent'},
                        {'label': 'No conversions', 'value': 'none'},
                    ],
                    value='fill_24_percent',
                    # Adjusted style to match height of input boxes (36px)
                    style={'fontSize': 16, 'height': '36px', 'lineHeight': '20px', 'textAlign': 'center'} 
                )
            ], style={'flex': '3', 'minWidth': '260px', 'textAlign': 'center'}),

            # Item 2G: IRMAA Strategy (Dropdown) - Added display: block to Label for consistency
            html.Div([
                html.Label("IRMAA Strategy", style={'fontWeight': 'bold', 'display': 'block', 'fontSize': 16, 'marginBottom': '6px'}),
                dcc.Dropdown(
                    id='irmaa_strategy',
                    options=[
                        {'label': 'Stay under Tier 0 ($218k)', 'value': 'fill_IRMAA_0'},
                        {'label': 'Stay under Tier 1 ($274k)', 'value': 'fill_IRMAA_1'},
                        {'label': 'Stay under Tier 2 ($342k)', 'value': 'fill_IRMAA_2'},
                        {'label': 'Stay under Tier 3 ($410k)', 'value': 'fill_IRMAA_3'},
                        {'label': 'Stay under Tier 4 ($750k)', 'value': 'fill_IRMAA_4'},
                    ],
                    value='fill_IRMAA_3',
                    # Adjusted style to match height of input boxes (36px)
                    style={'fontSize': 16, 'height': '36px', 'lineHeight': '20px', 'textAlign': 'center'}
                )
            ], style={'flex': '3', 'minWidth': '260px', 'textAlign': 'center'}),

        ], style={
            'display': 'flex',
            'gap': '15px 15px', # Ensures clean row and column separation
            'flexWrap': 'wrap',
            'marginBottom': '10px',
            'width': '100%',
            'boxSizing': 'border-box'
        }),

        # ----------------------------------------------------------------------
        # RESULTS SECTION 
        # ----------------------------------------------------------------------
        html.Div(
            style={
                'display': 'grid',
                'gridTemplateColumns': 'auto 1fr',
                'gap': '30px',
                'alignItems': 'center',
                'padding': '20px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '12px',
                'boxShadow': '0 4px 12px rgba(0,0,0,0.05)',
                'marginBottom': '15px',
                'minHeight': '68px'
            },
            children=[
                html.Div(
                    style={
                        'display': 'flex',
                        'gap': '30px',
                        'justifyContent': 'flex-start',
                        'flexWrap': 'nowrap',
                        'overflowX': 'auto',
                        'paddingRight': '20px'
                    },
                    children=[
                        html.Div(
                            pretty_currency_input('success-threshold', value=500000, label="Success Threshold")
                        ),
                        html.Div(
                            pretty_currency_input('avoid-ruin-threshold', value=500000, label="Avoid Ruin Threshold")
                        ),
                    ]
                ),
                html.Div(
                    id='success-header',
                    children="Click 'Run Simulation' to load results",
                    style={
                        'textAlign': 'center',
                        'fontWeight': 'bold',
                        'fontSize': '20px',
                        'color': '#1a1a1a',
                        'padding': '0px 10px',
                        'backgroundColor': '#e3f2fd',
                        'border': '2px solid #0052CC',
                        'borderRadius': '8px',
                        'minWidth': '340px',
                        'maxWidth': '800px',
                        'height': '50px',
                        'justifySelf': 'end',
                        'lineHeight': '1.3',
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'center'
                    }
                ),
            ]
        ),
        
 
        # The detailed plots section (Calls the function that now only returns plots)
        html.Div(id="results", children=[
            create_results_layout(),
        ]),

        # The detailed metrics table remains separate below the plots
        html.Div(
            id="metrics-table",
            children=html.P("Metrics will appear here after run.", style={"color": "#888", "fontStyle": "italic"}),
            style={"marginTop": "30px", "textAlign": "center"}
        ),
        
        # debug element
        html.Div(id="debug-output", style={"whiteSpace": "pre-wrap", "fontSize": 12, "display": "none"}),
        
    ] 
)
