# callbacks/results_callbacks.py

from dash import Input, Output, State, callback, no_update
import numpy as np
import plotly.graph_objects as go
from utils.plotting import generate_all_plots
from models import PlannerInputs

def register_results_callbacks(app):
    """
    This callback triggers whenever new simulation data is saved to the dcc.Store.
    It reads the stored results and regenerates all plots — cleanly and reliably.
    """

    @app.callback(
        Output("success-header", "children"),
        Output("portfolio-median", "figure"),
        Output("portfolio-p10", "figure"),
        Output("portfolio-p90", "figure"),
        Output("income-median", "figure"),
        Output("income-p10", "figure"),
        Output("income-p90", "figure"),
        Output("spending-median", "figure"),
        Output("spending-p10", "figure"),
        Output("spending-p90", "figure"),
        Output("magi", "figure"),
        Output("travel-gifting", "figure"),
        Output("roth-conversions", "figure"),
        Output("medicare-costs", "figure"),
        Output("metrics-table", "children"),
        Output("debug-output", "children"),

        Input("simulation-data-store", "data"),  # This is what triggers the update
        prevent_initial_call=True
    )
    def update_results_from_store(data_dict):
        if not data_dict:  # Nothing stored yet
            return no_update

        try:
            # === Reconstruct results with proper numpy arrays ===
            results = {
                "portfolio_paths": np.array(data_dict["portfolio_paths"]),
                "account_paths": {k: np.array(v) for k, v in data_dict.get("account_paths", {}).items()},
                "travel_paths": np.array(data_dict["travel"]),
                "gifting_paths": np.array(data_dict["gifting"]),
                "base_spending_paths": np.array(data_dict["base_spending"]),
                "lumpy_spending_paths": np.array(data_dict["lumpy"]),
                "rmd_paths": np.array(data_dict["rmds"]),
                "ssbenefit_paths": np.array(data_dict["ssbenefit"]),
                "portfolio_withdrawal_paths": np.array(data_dict["portfolio_withdrawal"]),
                "def457b_income_paths": np.array(data_dict["def457b_income"]),
                "pension_paths": np.array(data_dict["pension"]),
                "trust_income_paths": np.array(data_dict["trust_income"]),
                "taxes_paths": np.array(data_dict["taxes"]),
                "conversion_paths": np.array(data_dict["conversions"]),
                "medicare_paths": np.array(data_dict["medicare"]),
                "magi_paths": np.array(data_dict["magi"]),
                "success_rate": data_dict["success_rate"],
                "avoid_ruin_rate": data_dict["avoid_ruin_rate"],
                "final_median": data_dict.get("final_median", np.nan),
            }

            elapsed = data_dict.get("elapsed")
            inputs_dict = data_dict["inputs"]  # We stored this in simulation callback

            # Rebuild the PlannerInputs object
            inputs = PlannerInputs(**inputs_dict)

            # Generate plots
            all_figures, rate_header, metrics_table = generate_all_plots(results, inputs, elapsed)

            # Debug output
            debug_log = data_dict.get("debug_log", [])
            debug_text = "\n".join(map(str, debug_log))

            # Return in exact order expected by the outputs above
            return (
                rate_header,
                all_figures["portfolio-median"],
                all_figures["portfolio-p10"],
                all_figures["portfolio-p90"],
                all_figures["income-median"],
                all_figures["income-p10"],
                all_figures["income-p90"],
                all_figures["spending-median"],
                all_figures["spending-p10"],
                all_figures["spending-p90"],
                all_figures["magi"],
                all_figures["travel-gifting"],
                all_figures["roth-conversions"],
                all_figures["medicare-costs"],
                metrics_table,
                debug_text
            )

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print("Error in update_results_from_store:", tb)
            empty = go.Figure()
            error_msg = f"Plot update failed: {e}"
            return (
                "Error rendering results",
                empty, empty, empty,
                empty, empty, empty,
                empty, empty, empty,
                empty, empty, empty, empty,
                error_msg,
                tb
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
