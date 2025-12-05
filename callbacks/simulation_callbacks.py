# callbacks/simulation_callbacks.py

from dash import Input, Output, State, callback, no_update
from dash import html
import plotly.graph_objects as go
import time
import traceback
import numpy as np
from utils.input_adapter import get_planner_inputs
from engine.simulator import RetirementSimulator
from utils.plotting import generate_all_plots, get_figure_ids
from models import PlannerInputs
from utils.currency import clean_currency

years = None #initialize to pass to plotting

def register_simulation_callbacks(app):
    
    # Output for the main success/ruin rate header 
    SUCCESS_HEADER_OUTPUT = Output("success-header", "children") 

    # Dynamically build the list of Figure IDs + 2 metrics IDs
    FIGURE_OUTPUTS = [Output(id, "figure") for id in get_figure_ids()]

    # Output for use by register_returns_callback
    STORE_OUTPUTS = Output('simulation-data-store', 'data')
    
    @app.callback(
        # Header
        SUCCESS_HEADER_OUTPUT,
        # Figures
        *FIGURE_OUTPUTS,
        # Metrics Table 
        Output("metrics-table", "children"),
        # Debug outputs
        Output("debug-output", "children"),
        # Store output
        STORE_OUTPUTS,
        
        # Input
        Input("run", "n_clicks"),

        State("nsims", "value"), 
        State("base_annual_spending", "value"),
        State("withdrawal_rate", "value"),
        State("portfolio-grid", "rowData"),
        State("max_roth", "value"),
        State("travel", "value"),
        State("gifting", "value"),
        State("tax_strategy", "value"),
        State("irmaa_strategy", "value"),
        prevent_initial_call=True
    )
    def run_simulation(n_clicks, n_sims, base_spending, withdrawal_rate, portfolio_rows, max_roth, travel, gifting, tax_strategy, irmaa_strategy):
        from models import PlannerInputs
        if not n_clicks:
            return no_update

        start_time = time.time()

        try:
            # Build inputs from current UI
            inputs = get_planner_inputs(
                num_simulations=n_sims,
                base_annual_spending=base_spending,
                withdrawal_rate=withdrawal_rate,
                portfolio_data=portfolio_rows,
                max_roth=max_roth,
                travel=travel,
                gifting=gifting,
                tax_strategy=tax_strategy,
                irmaa_strategy=irmaa_strategy
            )

            sim = RetirementSimulator(inputs)
            results = sim.run_simulation()

            elapsed = time.time() - start_time

            # Build years array used in plots (needs to go back 2 years for IRMAA calcs)
            current_age_p1 = inputs.current_year - inputs.person1_birth_year
            current_age_p2 = inputs.current_year - inputs.person2_birth_year
            min_age = min(current_age_p1, current_age_p2)
            n_years = inputs.end_age - min_age
            global years
            years = np.arange(inputs.current_year - 2, inputs.current_year + n_years) 

            # ----------------------------------------------------------------
            # GENERATE PLOTS
            # ----------------------------------------------------------------
            all_figures, rate_header, metrics_table = generate_all_plots(results, inputs, elapsed)
            
            # Prepare the figure list in the correct order for the return statement
            figure_list = [all_figures[id] for id in get_figure_ids()]
            
            # ----------------------------------------------------------------
            # PREPARE DATA STORE FOR RESULTS
            # ----------------------------------------------------------------
            results_to_store = {
                "portfolio_paths": results["portfolio_paths"].tolist(),
                "account_paths": {k: v.tolist() for k, v in results.get("account_paths", {}).items()},
                "travel": results.get("travel_paths", np.zeros((1,1))).tolist(),
                "gifting": results.get("gifting_paths", np.zeros((1,1))).tolist(),
                "base_spending": results.get("base_spending_paths", np.zeros((1,1))).tolist(),
                "lumpy": results.get("lumpy_spending_paths", np.zeros((1,1))).tolist(),
                "rmds": results.get("rmd_paths", np.zeros((1,1))).tolist(),
                "ssbenefit": results.get("ssbenefit_paths", np.zeros((1,1))).tolist(),
                "portfolio_withdrawal": results.get("portfolio_withdrawal_paths", np.zeros((1,1))).tolist(),
                "def457b_income": results.get("def457b_income_paths", np.zeros((1,1))).tolist(),
                "pension": results.get("pension_paths", np.zeros((1,1))).tolist(),
                "trust_income": results.get("trust_income_paths", np.zeros((1,1))).tolist(),
                "taxes": results.get("taxes_paths", np.zeros((1,1))).tolist(),
                "conversions": results.get("conversion_paths", np.zeros((1,1))).tolist(),
                "medicare": results.get("medicare_paths", np.zeros((1,1))).tolist(),
                "magi": results.get("magi_paths", np.zeros((1,1))).tolist(),
                "success_rate": float(results.get("success_rate", 0)),
                "avoid_ruin_rate": float(results.get("avoid_ruin_rate", 0)),
                "final_median": float(results.get("final_median", 0)),
                "elapsed": elapsed,
                "debug_log": getattr(sim, "debug_log", []),
                # Store inputs too if needed later
                #"inputs": inputs.dict() if hasattr(inputs, "dict") else inputs.__dict__,
            }
 
            # ----------------------------------------------------------------
            # Metrics and Debug 
            # ----------------------------------------------------------------
            success = results.get("success_rate", 0)
            avoid_ruin = results.get("avoid_ruin_rate", 0)
            sim_count = getattr(inputs, "num_simulations", 1000)

            final_debug_content = (
                f"Simulation completed in {elapsed:.2f}s\n"
                f"Success: {success:.1f} | Avoid Ruin: {avoid_ruin:.1f}\n"
                f"Debug log:\n" + "\n".join(getattr(sim, "debug_log", []))
            )
            debug_output = html.Div(
                final_debug_content,
                style={'whiteSpace': 'pre-wrap', 'fontSize': '18px', 'color': 'blue', 'marginTop': '10px'}
            )
            
            # The return must match the order of *FIGURE_OUTPUTS, then the two metrics
            return rate_header, *figure_list, metrics_table, debug_output, results_to_store

        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            error_msg = f"Simulation failed: {str(e)}\nFull traceback:\n{tb}"
            
            # Return empty figures and the error message to all outputs
            error_header = html.Div("Simulation Failed", style={'color': 'red', 'fontSize': '40px'})
            empty_figures = [go.Figure() for _ in get_figure_ids()] # Now 13 figures
            error_div = html.Div(error_msg, style={"color": "red"})

            error_div = html.Div(error_msg, style={"color": "red"})
            
            return error_header, *empty_figures, error_div, error_msg, results_to_store
