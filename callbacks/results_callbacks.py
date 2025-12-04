# results_callbacks.py
from dash import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def register_results_callbacks(app, sim, inputs, res, data_dict):
    """
    data_dict contains precomputed arrays needed for plots:
    - years, account_paths, income trajectories, spending plans, MAGI, medicare, conversions, etc.
    """

    @app.callback(
        Output('success-header', 'children'),
        Output('sim-info', 'children'),
        Output('portfolio-median', 'figure'),
        Output('portfolio-p10', 'figure'),
        Output('portfolio-p90', 'figure'),
        Output('income-median', 'figure'),
        Output('income-p10', 'figure'),
        Output('income-p90', 'figure'),
        Output('spending-median', 'figure'),
        Output('spending-p10', 'figure'),
        Output('spending-p90', 'figure'),
        Output('magi', 'figure'),
        Output('travel-gifting', 'figure'),
        Output('roth-conversions', 'figure'),
        Output('medicare-costs', 'figure'),
        Output('debug-output', 'children'),
        Input('run-button', 'n_clicks')
    )

def update_results(n_clicks):
        # Just unpack what the simulator already gave us
        results = {
            "portfolio_paths": data_dict['portfolio_paths'],
            "account_paths": data_dict['account_paths'],
            "travel_paths": data_dict['travel'],
            "gifting_paths": data_dict['gifting'],
            "base_spending_paths": data_dict['base_spending'],
            "lumpy_spending_paths": data_dict['lumpy'],
            "rmd_paths": data_dict['rmds'],
            "ssbenefit_paths": data_dict['ssbenefit'],
            "portfolio_withdrawal_paths": data_dict['portfolio_withdrawal'],
            "def457b_income_paths": data_dict['def457b_income'],
            "pension_paths": data_dict['pension'],
            "trust_income_paths": data_dict['trust_income'],
            "taxes_paths": data_dict['taxes'],
            "conversion_paths": data_dict['conversions'],
            "medicare_paths": data_dict['medicare'],
            "magi_paths": data_dict['magi'],
            "success_rate": data_dict['success_rate'],
            "avoid_ruin_rate": data_dict['avoid_ruin_rate'],
        }

        # Build the header exactly as you want

        print("DEBUG: success_rate =", data_dict['success_rate'])
        print("DEBUG: vanguard_color result =", vanguard_color(data_dict['success_rate']))

        success_header = html.Div(
            [
                html.Span(f"Success Rate: {data_dict['success_rate']:.1f}%", 
                          style={'color': vanguard_color(data_dict['success_rate']), 'fontSize': 36, 'fontWeight': 'bold'}),
                "   •   ",
                html.Span(f"Ruin Avoidance: {data_dict['avoid_ruin_rate']:.1f}%", 
                          style={'color': '#0052CC', 'fontSize': 30}),
            ],
            style={
                'textAlign': 'center',
                'padding': '20px',
                'backgroundColor': '#f8f9fa',
                'borderBottom': '3px solid #e9ecef',
                'marginBottom': '20px'
            }
        )

        # NOW USE YOUR UTILITY — this does everything else perfectly
        all_figures, _, _ = generate_all_plots(results, inputs, data_dict['elapsed'])

        sim_info = f"{data_dict['nsims']:,} simulations • {data_dict['elapsed']:.1f}s"
        debug_text = "\n".join(sim.debug_log) if hasattr(sim, 'debug_log') else ""

        return (
            success_header,
            sim_info,
            all_figures["portfolio-median"], all_figures["portfolio-p10"], all_figures["portfolio-p90"],
            all_figures["income-median"],    all_figures["income-p10"],    all_figures["income-p90"],
            all_figures["spending-median"],   all_figures["spending-p10"],   all_figures["spending-p90"],
            all_figures["magi"], all_figures["travel-gifting"],
            all_figures["roth-conversions"], all_figures["medicare-costs"],
            debug_text
        )

def vanguard_color(success_rate):
    sr = max(0, min(100, success_rate))
    if sr >= 75:
        r = int(255 * (1 - (sr - 75)/25)); g = 200; b = 0
    elif sr >= 50:
        r = 255; g = int(200 * ((sr - 50)/25 + 0.5)); b = 0
    else:
        r = 255; g = int(200 * (sr/50)); b = 0
    return f'rgb({r},{g},{b})'
