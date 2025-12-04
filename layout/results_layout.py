# results_layout.py
from dash import html, dcc

def create_results_layout():
    """Static layout with placeholders for dynamic graphs."""
    return html.Div([
        html.Div(
            id='success-header',  # ← changed from 'success-rate' to match your callback
            style={
                'padding': '30px 20px',
                'backgroundColor': '#f8f9fa',
                'borderBottom': '5px solid #0052CC',
                'textAlign': 'center',
                'fontWeight': 'bold',
                'fontSize': '28px',
                'minHeight': '100px',           # prevents collapse when empty
                'position': 'relative',
                'zIndex': 9999,
                'marginBottom': '40px'
            }
        ),

        # Sim info line
        html.P(id='sim-info', style={
            'textAlign': 'center',
            'color': '#7f8c8d',
            'fontSize': '18px',
            'marginTop': '-20px',
            'marginBottom': '40px'
        }),

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

        # Debug output
        html.Pre(id='debug-output')
    ], style={'maxWidth': '1400px', 'margin': '0 auto'})
