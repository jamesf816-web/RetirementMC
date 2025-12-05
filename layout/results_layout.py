# results_layout.py
from dash import html, dcc

def create_results_layout():
    """
    Static layout for the plots section only.
    The success-header and sim-info elements are moved to main_layout.py.
    """
    return html.Div([

        # NOTE: success-header and sim-info removed from here and placed in main_layout.py
        
        # Portfolio trajectories
        html.Div([dcc.Graph(id='portfolio-median')], style={'margin': '40px 0'}),
        html.Div([dcc.Graph(id='portfolio-p10')], style={'margin': '40px 0'}),
        html.Div([dcc.Graph(id='portfolio-p90')], style={'margin': '40px 0'}),

        # Income trajectories
        html.Div([dcc.Graph(id='income-median')], style={'margin': '40px 0'}),
        html.Div([dcc.Graph(id='income-p10')], style={'margin': '40px 0'}),
        html.Div([dcc.Graph(id='income-p90')], style={'margin': '40px 0'}),

        # Spending trajectories
        html.Div([dcc.Graph(id='spending-median')], style={'margin': '40px 0'}),
        html.Div([dcc.Graph(id='spending-p10')], style={'margin': '40px 0'}),
        html.Div([dcc.Graph(id='spending-p90')], style={'margin': '40px 0'}),

        # MAGI, travel/gifting, ROTH conversions, Medicare costs
        html.Div([dcc.Graph(id='magi')], style={'margin': '40px 0'}),
        html.Div([dcc.Graph(id='travel-gifting')], style={'margin': '40px 0'}),
        html.Div([dcc.Graph(id='roth-conversions')], style={'margin': '40px 0'}),
        html.Div([dcc.Graph(id='medicare-costs')], style={'margin': '40px 0'}),

        # NOTE: Metrics table abd debug log removed from here and placed in main_layout.py

    ], style={'maxWidth': '1400px', 'margin': '0 auto'})
