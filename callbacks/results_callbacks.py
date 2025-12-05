from dash import Input, Output
import numpy as np
from .utils.plotting import generate_all_plots

def register_results_callbacks(app, sim, inputs, res, data_dict):
    """
    Registers the callback for updating simulation results.
    """

    @app.callback(
        Output('success-header', 'children'),
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
        Output('metrics-table', 'children'),
        Output('debug-output', 'children'),
        Input('run', 'n_clicks')
    )
    def update_results(n_clicks):
        # Ensure data_dict exists and contains all expected arrays
        results = {
            "portfolio_paths": data_dict.get('portfolio_paths', np.zeros((1,1))),
            "account_paths": data_dict.get('account_paths', {}),
            "travel_paths": data_dict.get('travel', np.zeros((1,1))),
            "gifting_paths": data_dict.get('gifting', np.zeros((1,1))),
            "base_spending_paths": data_dict.get('base_spending', np.zeros((1,1))),
            "lumpy_spending_paths": data_dict.get('lumpy', np.zeros((1,1))),
            "rmd_paths": data_dict.get('rmds', np.zeros((1,1))),
            "ssbenefit_paths": data_dict.get('ssbenefit', np.zeros((1,1))),
            "portfolio_withdrawal_paths": data_dict.get('portfolio_withdrawal', np.zeros((1,1))),
            "def457b_income_paths": data_dict.get('def457b_income', np.zeros((1,1))),
            "pension_paths": data_dict.get('pension', np.zeros((1,1))),
            "trust_income_paths": data_dict.get('trust_income', np.zeros((1,1))),
            "taxes_paths": data_dict.get('taxes', np.zeros((1,1))),
            "conversion_paths": data_dict.get('conversions', np.zeros((1,1))),
            "medicare_paths": data_dict.get('medicare', np.zeros((1,1))),
            "magi_paths": data_dict.get('magi', np.zeros((1,1))),
            "success_rate": data_dict.get('success_rate', 0.0),
            "avoid_ruin_rate": data_dict.get('avoid_ruin_rate', 0.0),
            "success_header": data_dict.get('success_header', "Click 'Run Simulation' to load results"),
            "final_median": data_dict.get('final_median', np.nan)
        }

        elapsed = data_dict.get("elapsed", None)

        # Generate all figures, headers, and metrics table
        all_figures, rate_header, metrics_table = generate_all_plots(results, inputs, elapsed)

        debug_log = data_dict.get("debug_log", [])
        debug_text = "\n".join(str(x) for x in debug_log)

        return (
            rate_header,
            all_figures["portfolio-median"], all_figures["portfolio-p10"], all_figures["portfolio-p90"],
            all_figures["income-median"], all_figures["income-p10"], all_figures["income-p90"],
            all_figures["spending-median"], all_figures["spending-p10"], all_figures["spending-p90"],
            all_figures["magi"], all_figures["travel-gifting"],
            all_figures["roth-conversions"], all_figures["medicare-costs"],
            metrics_table,
            debug_text
        )


def vanguard_color(success_rate):
    # Check for valid numeric input. If not, return a safe, default color.
    if not isinstance(success_rate, (int, float)) or success_rate is None or np.isnan(success_rate):
        print(f"ERROR: vanguard_color received invalid rate: {success_rate}")
        return 'rgb(128, 128, 128)'  # Gray color for safety

    sr = max(0, min(100, success_rate))
    
    if sr >= 75:
        r = int(255 * (1 - (sr - 75)/25)); g = 200; b = 0
    elif sr >= 50:
        r = 255; g = int(200 * ((sr - 50)/25 + 0.5)); b = 0
    else:
        r = 255; g = int(200 * (sr/50)); b = 0
        
    return f'rgb({int(r)},{int(g)},{int(b)})'
