from dash import Input, Output, State, ctx, html
from dash.exceptions import PreventUpdate
from engine.simulator import RetirementSimulator
from callbacks.results_callbacks import register_callbacks
import time

def register_simulation_callbacks(app, get_planner_inputs):
    """
    Registers the main simulation callback and connects to the plotting callback
    in results_callbacks.py. Handles multiple re-runs safely
    """
    @app.callback(
        Output('results-container', 'children'),  # placeholder, figures update in results_callbacks.py
        Output('debug-output', 'children'),
        Input('run', 'n_clicks'),
        State('portfolio-store', 'data'),
        State('setup-store', 'data'),
        prevent_initial_call=True
    )
    def run_simulation(run, portfolio, setup):
        if ctx.triggered_id != 'run' or not run:
            raise PreventUpdate

        t0 = time.time()

        # Build typed inputs from store
        inputs = get_planner_inputs(portfolio_data=portfolio, setup_data=setup)

        # Run simulation
        sim = RetirementSimulator(inputs)
        res = sim.run_simulation()
        elapsed = time.time() - t0

        # Precompute all arrays needed for results_callbacks
        data_dict = {
            'years': res['years'],
            'account_paths': res['account_paths'],
            'income_trajs': res['income_trajs'],
            'spending_trajs': res['spending_trajs'],
            'magi': res['magi'],
            'travel': res['travel'],
            'gifting': res['gifting'],
            'conversions': res['conversions'],
            'medicare': res['medicare'],
            'nsims': inputs.nsims,
            'success_rate': res.get('success_rate', 0),
            'avoid_ruin_rate': res.get('avoid_ruin_rate', 0),
            'elapsed': elapsed,
            'tax_type_map': {name: acc['tax'] for name, acc in inputs.accounts.items()}
        }

        # Re-register plot callbacks safely (Dash allows multiple registrations if done carefully)
        # Using try/except to avoid duplicate callback errors
        try:
            register_callbacks(app, sim, inputs, res, data_dict)
        except Exception as e:
            # Optional: log or ignore duplicate registration
            print(f"Callback registration skipped (already registered): {e}")

        # Build richer placeholder layout
        results_layout = html.Div([
            html.H3("Simulation Complete", style={'textAlign': 'center'}),
            html.P(f"{inputs.nsims:,} simulations in {elapsed:.2f}s", style={'textAlign': 'center'}),
            html.P(f"Success Rate: {res.get('success_rate', 0):.2f}%  |  Avoid Ruin: {res.get('avoid_ruin_rate', 0):.2f}%",
                   style={'textAlign': 'center', 'fontWeight': 'bold'})
        ], style={'margin': '20px auto', 'maxWidth': '800px'})

        # Return placeholder and debug log
        debug_text = "\n".join(sim.debug_log)

        return results_layout, debug_text

