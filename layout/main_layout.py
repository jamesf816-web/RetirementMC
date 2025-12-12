# main_layout.py
import dash
from dash import Dash
from dash import dcc
from dash import html

import dash_ag_grid as dag
import plotly.graph_objects as go
import numpy as np

from utils.xml_loader import DEFAULT_SETUP, DEFAULT_ACCOUNTS
from utils.currency import pretty_currency_input, pretty_year_input, pretty_percent_input

from layout.results_layout import create_results_layout

from callbacks.editor_callbacks import format_currency_output

# Calculate the initial total balance from the default data
INITIAL_TOTAL_BALANCE = sum(v.get('balance', 0) for v in DEFAULT_ACCOUNTS.values())
INITIAL_TOTAL_BALANCE_STR = format_currency_output(INITIAL_TOTAL_BALANCE)

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
            # COLUMN 1A: Collapsible Portfolio Editor Button
            html.Button(
                "Portfolio Editor – Click to Open",
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
                }
            ),
         
            # COLUMN 1B: Slider (takes up most of the space)
            html.Div([
                html.Label("Number of Simulations", style={'fontSize': 16, 'fontWeight': 'bold'}),
                dcc.Slider(
                    id='nsims', min=1, max=30000, step=100, value=1000,
                    marks={i: f"{i//1000}k" for i in range(0, 31000, 2000)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], style={'flex': 1, 'minWidth': '300px', 'padding': '0 20px'}
            ), 

            # COLUMN 1C: Run Button 
            html.Button(
                "Run Simulation",
                id="run",
                n_clicks=0,
                style={
                    'padding': '12px 20px',
                    'fontSize': '16px',
                    'fontWeight': 'bold',
                    'backgroundColor': '#3498db',
                    'color': 'white', 
                    'border': 'none',
                    'borderRadius': '8px',
                    'cursor': 'pointer',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                    'whiteSpace': 'nowrap',
                    'height': '50px',
                    'alignSelf': 'flex-end',
                }
            ),
            
        ], style={
            'display': 'flex', 
            'alignItems': 'flex-end', 
            'gap': '15px', 
            'marginBottom': '30px', 
            'flexWrap': 'wrap', 
            'width': '100%',
            'boxSizing': 'border-box'
        }),
  
        # ----------------------------------------------------------------------
        # ROW 1 - COLLAPSIBLE PORTFOLIO EDITOR FULL WIDTH WHEN OPEN
        # ----------------------------------------------------------------------
        dcc.Download(id="download-xml-portfolio"),

        html.Div(
            id="portfolio-collapse-content",
            style={'display': 'none'},
            children=[
                # Outer Div to apply styling (border, background, padding) to the entire editor content
                html.Div(
                    style={
                        'padding': '25px',
                        'border': '2px solid #ddd',
                        'borderRadius': '12px',
                        'backgroundColor': '#fff',
                        'boxShadow': '0 8px 25px rgba(0,0,0,0.1)',
                        'marginBottom': '30px'
                    },
                    children=[
                        # Div for the buttons row (1. File/2. Save/3. Add/4. Reset/5. Status)
                        html.Div([
                            # 1. Choose XML File (dcc.Upload with inline button)
                            dcc.Upload(
                                id='upload-data',
                                children=html.Button(
                                    "Choose XML File",
                                    style={
                                        'marginRight': '10px',
                                        'backgroundColor': '#f39c12', # Orange for file selection
                                        'color': 'white',
                                        'border': 'none',
                                        'padding': '10px 20px',
                                        'borderRadius': '4px',
                                        'cursor': 'pointer'
                                    }
                                ),
                                multiple=False,
                                accept=".xml",
                                style={'display': 'inline-block', 'marginRight': '10px'}
                            ),

                            # 2. Save to XML Button
                            html.Button(
                                "Save Portfolio to XML",
                                id="save-portfolio-btn",
                                n_clicks=0,
                                style={
                                    'marginRight': '30px', # Add extra space after Save button
                                    'backgroundColor': '#2ecc71', # Green for saving
                                    'color': 'white',
                                    'border': 'none',
                                    'padding': '10px 20px',
                                    'borderRadius': '4px',
                                    'cursor': 'pointer'
                                }
                            ),

                           # 2b. Save-As functionality
                            html.Span("Save as: ", style={'margin': '0 10px', 'fontWeight': 'bold'}),
                            dcc.Input(
                                id='save-filename-input',
                                type='text',
                                placeholder='my_portfolio.xml',
                                value='default_portfolio.xml',  # Will be filled automatically if a new file is uploaded
                                style={
                                    'width': '200px',
                                    'padding': '8px',
                                    'borderRadius': '4px',
                                    'border': '1px solid #ccc',
                                    'fontSize': '14px',
                                    'marginRight': '20px'
                                }
                            ),

                            # 3. Add New Account
                            html.Button("Add New Account", id="add-account-btn", n_clicks=0,
                                         style={'marginRight': '10px', 'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'borderRadius': '4px'}),
                            
                            # 4. Reset to Defaults
                            html.Button("Reset to Defaults", id="reset-portfolio-btn", n_clicks=0,
                                         style={'backgroundColor': 'blue', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'borderRadius': '4px'}),

                            # 5. Total Portfolio Balance Display
                            html.Div(
                                id="total-portfolio-balance",
                                children=f"Total Portfolio Balance: {INITIAL_TOTAL_BALANCE_STR}",
                                style={
                                    'marginLeft': 'auto',
                                    'marginRight': '30px',
                                    'fontWeight': 'bold',
                                    'fontSize': '20px',
                                    'color': '#34495e',
                                    'minWidth': '250px', 
                                    'textAlign': 'right'
                                }
                            ),

                            # 6. Portfolio Status Div
                            html.Div(id="portfolio-status", style={'display': 'inline-block', 'marginLeft': '20px', 'color': '#7f8c8d', 'fontSize': '14px'}),

                            # 7. Delete Account
                            html.Button("Delete Account", id="delete-selected-btn", n_clicks=0,
                                        style={'backgroundColor': 'red', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'borderRadius': '4px'}),
                                 
                        ], style={'marginBottom': '15px', 'display': 'flex', 'alignItems': 'center'}),


                        # dag.AgGrid for Portfolio Editor
                        dag.AgGrid(
                            id='portfolio-grid',
                            columnDefs=[
                                {"field": "name", "headerName": "Account Name", "editable": True, "pinned": "left", "width": 180},
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
                                 "cellEditorParams": {"values": ["taxable", "traditional", "roth", "inherited", "trust"]}, "width": 130,
                                "editable": True},                          
                                {"field": "owner", "headerName": "Owner", "cellEditor": "agSelectCellEditor", "editable": True,    
                                 "cellEditorParams": {"values": ["person1", "person2"]}, "width": 110},
                                {"field": "basis", "headerName": "Basis ($)",
                                 "valueFormatter": {"function": "params.value == null ? '' : '$' + Number(params.value).toLocaleString()"},
                                 "editable": True},
                                {"field": "income", "headerName": "Mand. Yield", "editable": True, "width": 110},
                                {"field": "rmd_factor_table", "headerName": "RMD Table", "editable": True, "width": 130},
                                {
                                    "headerName": "Select to Delete",
                                    "checkboxSelection": True,
                                    "headerCheckboxSelection": True,
                                    "width": 90,
                                    "pinned": "right",
                                    "sortable": False,
                                    "filter": False
                                },                              

                            ],
                            rowData=[{**v, "name": k, "delete": False} for k, v in DEFAULT_ACCOUNTS.items()],
                            defaultColDef={
                                "flex": 1,
                                "minWidth": 100,
                                "resizable": True,
                                "sortable": True,
                                "filter": True,
                                "floatingFilter": True
                            },
                            dashGridOptions={"rowHeight": 48, "animateRows": False, "onCellValueChanged": {"function": "console.log('changed')"}},
                            style={"height": 550, "minWidth": 1200},
                            className="ag-theme-alpine",
                            
                        )
                    ] 
                )
            ]
        ),

        # ----------------------------------------------------------------------
        # ROW 2: Main Inputs and Dropdowns 
        # ----------------------------------------------------------------------
        html.Div([
            # Item 2A: Base Spending - Correct call: pretty_currency_input returns a list of children
            html.Div(
                pretty_currency_input('base_annual_spending', value=140000, label="Base Spending"),
                style={'flex': '1', 'minWidth': '150px', 'textAlign': 'center'}
            ),

            # Item 2B: Withdrawal Rate - Now using the new pretty_percent_input for matching style
            html.Div(
                pretty_percent_input('withdrawal_rate', value=0.050, step=0.001, label="Withdrawal Rate", placeholder="5.0%", decimals=2),
                style={'flex': '1', 'minWidth': '150px', 'textAlign': 'center'}
            ),

            # Item 2C: Max Roth Conv
            html.Div(
                pretty_currency_input('max_roth', value=160000, label="Max Roth Conv"),
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

            # Item 2F: Success Threshold
            html.Div(
                pretty_currency_input('success_threshold', value=float(100_000), label="Success Threshold"),
                style={'flex': '1', 'minWidth': '150px', 'textAlign': 'center'}
            ),
            
            # Item 2G: Avoid Ruin Threshold
            html.Div(
                pretty_currency_input('avoid_ruin_threshold', value=float(50_000), label="Avoid Ruin"),
                style={'flex': '1', 'minWidth': '150px', 'textAlign': 'center'}      
            ),

        ], style={
            'display': 'flex',
            'gap': '15px 15px', # Ensures clean row and column separation
            'flexWrap': 'wrap',
            'marginBottom': '15px',
            'width': '100%',
            'boxSizing': 'border-box'
        }),

        # ----------------------------------------------------------------------
        #  ROW 3:Remaining Inputs + Success/Avoid Ruin Output
        # ----------------------------------------------------------------------
        html.Div([
            # Item 3A: ROTH Tax Strategy (Dropdown) - Added display: block to Label for consistency
            html.Div([
                html.Label("ROTH Tax Bracket", style={'fontWeight': 'bold', 'display': 'block', 'fontSize': 16, 'marginBottom': '6px'}),
                dcc.Dropdown(
                    id='roth_tax_bracket',
                    options=[
                        {'label': 'Fill 22% bracket ($211k)', 'value': 'fill_22_percent'},
                        {'label': 'Fill 24% bracket ($404k)', 'value': 'fill_24_percent'},
                        {'label': 'Fill 32% bracket ($512k)', 'value': 'fill_32_percent'},
                        {'label': 'Fill 35% bracket ($767k)', 'value': 'fill_35_percent'},
                        {'label': 'No conversions', 'value': 'none'},
                    ],
                    value='fill_24_percent',
                    # Adjusted style to match height of input boxes (36px)
                    style={'fontSize': 16, 'height': '40px', 'lineHeight': '15px', 'textAlign': 'center'} 
                )
            ], style={'flex': '1', 'minWidth': '200px', 'textAlign': 'center'}),

            # Item 3B: ROTH IRMAA Strategy (Dropdown) - Added display: block to Label for consistency
            html.Div([
                html.Label("ROTH IRMAA Threshold", style={'fontWeight': 'bold', 'display': 'block', 'fontSize': 16, 'marginBottom': '6px'}),
                dcc.Dropdown(
                    id='roth_irmaa_threshold',
                    options=[
                        {'label': 'Stay under Tier 1 ($218k)', 'value': 'fill_IRMAA_1'},
                        {'label': 'Stay under Tier 2 ($274k)', 'value': 'fill_IRMAA_2'},
                        {'label': 'Stay under Tier 3 ($342k)', 'value': 'fill_IRMAA_3'},
                        {'label': 'Stay under Tier 4 ($410k)', 'value': 'fill_IRMAA_4'},
                        {'label': 'Stay under Tier 5 ($750k)', 'value': 'fill_IRMAA_5'},
                    ],
                    value='fill_IRMAA_4',
                    # Adjusted style to match height of input boxes (36px)
                    style={'fontSize': 16, 'height': '40px', 'lineHeight': '15px', 'textAlign': 'center'}
                )
            ], style={'flex': '1', 'minWidth': '210px', 'textAlign': 'center'}),

            # Item 3C: Tax Strategy (Dropdown) - Added display: block to Label for consistency
            html.Div([
                html.Label("Tax Strategy", style={'fontWeight': 'bold', 'display': 'block', 'fontSize': 16, 'marginBottom': '6px'}),
                dcc.Dropdown(
                    id='tax_strategy',
                    options=[
                        {'label': 'Typical (Taxable - Trusts - Traditional - Roth)', 'value': 'typical'},
                        {'label': 'Trusts First (Trusts - Taxable - Traditional - Roth)', 'value': 'trust_first'},
                        {'label': 'Trusts Last (Taxable - Traditional - Roth - Trusts)', 'value': 'preserve_trust'},
                        {'label': 'Minimize RMDs (Traditional - Taxable - Trusts - Roth)', 'value': 'lower_rmds'},
                        {'label': 'Default (Traditional - Taxable - Roth - Trusts)', 'value': 'default'},
                    ],
                    value='typical',
                    # Adjusted style to match height of input boxes (36px)
                    style={'fontSize': 16, 'height': '40px', 'lineHeight': '15px', 'textAlign': 'center'}
                )
            ], style={'flex': '1', 'minWidth': '250px', 'textAlign': 'center'}),

            # Item 3D: SS Trust Fund Failure Year
            html.Div(
                pretty_year_input('ss_fail_year', value=int(2099), label="SS Fail Year"),
                style={'flex': '1', 'minWidth': '120px', 'textAlign': 'center'}
            ),
            
            # Item 3E:  SS Trust Fund % reduction in benefits
            html.Div(
                pretty_percent_input('ss_fail_percent', value=float(0.23), label="SS Redution", decimals=1),
                style={'flex': '1', 'minWidth': '120px', 'textAlign': 'center'}
            ),
            
            # Item 3F: Output of Success/Avoid Failure Rates
            html.Div(
                id='success_header',
                children="Click 'Run Simulation' to load results",
                style={'flex': '3',
                    'textAlign': 'center',
                    'fontWeight': 'bold',
                    'fontSize': '20px',
                    'color': '#1a1a1a',
                    'padding': '0px 10px',
                    'backgroundColor': '#e3f2fd',
                    'border': '2px solid #0052CC',
                    'borderRadius': '8px',
                    'minWidth': "250px",
                    'height': '60px',
                    'justifySelf': 'end',
                    'lineHeight': '1.2',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center'
                }
            ),
            
        ], style={
            'display': 'flex',
            'gap': '15px 15px', # Ensures clean row and column separation
            'flexWrap': 'wrap',
            'marginBottom': '10px',
            'width': '100%',
            'boxSizing': 'border-box'
        }),
        
        # ----------------------------------------------------------------------
        # PLOTS SECTION 
        # ----------------------------------------------------------------------
        html.Div(id="results", children=[
            create_results_layout(),
        ]),
        
        # ----------------------------------------------------------------------
        # DEBUG LOG
        # ----------------------------------------------------------------------
        html.Div(id="debug-output", style={"whiteSpace": "pre-wrap", "fontSize": 12, "display": "none"}),
        
    ] 
)

