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
        Output('success-rate', 'children'),
        Output('success-rate', 'style'),
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
        # Unpack data
        years = data_dict['years']
        account_paths = data_dict['account_paths']
        income_trajs = data_dict['income_trajs']
        spending_trajs = data_dict['spending_trajs']
        magi = data_dict['magi']
        travel = data_dict['travel']
        gifting = data_dict['gifting']
        conversions = data_dict['conversions']
        medicare = data_dict['medicare']
        nsims = data_dict['nsims']
        elapsed = data_dict['elapsed']
        success_rate = data_dict['success_rate']
        avoid_ruin_rate = data_dict['avoid_ruin_rate']

        colors = px.colors.qualitative.Vivid
        px_colors = px.colors.qualitative.Plotly

        # --- Portfolio Figures ---
        fig_port_median = go.Figure()
        fig_port_p10 = go.Figure()
        fig_port_p90 = go.Figure()
        tax_type_map = data_dict['tax_type_map']

        for i, (tax, label) in enumerate(tax_type_map.items()):
            paths = np.zeros_like(account_paths[list(account_paths.keys())[0]])
            for name, traj in account_paths.items():
                if inputs.accounts[name]['tax'] == tax:
                    paths += traj
            if paths.max() == 0:
                continue
            median = np.median(paths, axis=0)
            p10 = np.percentile(paths, 10, axis=0)
            p90 = np.percentile(paths, 90, axis=0)
            fig_port_median.add_trace(go.Scatter(x=years[1:], y=median[1:], stackgroup='one', fillcolor=colors[i % len(colors)], name=tax))
            fig_port_p10.add_trace(go.Scatter(x=years[1:], y=p10[1:], stackgroup='one', fillcolor=colors[i % len(colors)], name=tax))
            fig_port_p90.add_trace(go.Scatter(x=years[1:], y=p90[1:], stackgroup='one', fillcolor=colors[i % len(colors)], name=tax))

        for f, title in zip([fig_port_median, fig_port_p10, fig_port_p90],
                            ["Portfolio Median", "Portfolio 10% CL", "Portfolio 90% CL"]):
            f.update_layout(title=title, xaxis_title="Year", yaxis_title="Balance ($)", template="plotly_white", hovermode="x unified", height=400)

        # --- Income Figures ---
        fig_income_median = go.Figure()
        fig_income_p10 = go.Figure()
        fig_income_p90 = go.Figure()
        for i, (name, traj) in enumerate(income_trajs.items()):
            median = np.median(traj, axis=0)
            p10 = np.percentile(traj, 10, axis=0)
            p90 = np.percentile(traj, 90, axis=0)
            fig_income_median.add_trace(go.Scatter(x=years[2:], y=median[2:], stackgroup='one', fillcolor=px_colors[i % len(px_colors)], name=name))
            fig_income_p10.add_trace(go.Scatter(x=years[2:], y=p10[2:], stackgroup='one', fillcolor=px_colors[i % len(px_colors)], name=name))
            fig_income_p90.add_trace(go.Scatter(x=years[2:], y=p90[2:], stackgroup='one', fillcolor=px_colors[i % len(px_colors)], name=name))
        for f, title in zip([fig_income_median, fig_income_p10, fig_income_p90],
                            ["Income Median", "Income 10% CL", "Income 90% CL"]):
            f.update_layout(title=title, xaxis_title="Year", yaxis_title="Income ($)", template="plotly_white", hovermode="x unified", height=400)

        # --- Spending Figures ---
        fig_spend_median = go.Figure()
        fig_spend_p10 = go.Figure()
        fig_spend_p90 = go.Figure()
        for i, (name, traj) in enumerate(spending_trajs.items()):
            median = np.median(traj, axis=0)
            p10 = np.percentile(traj, 10, axis=0)
            p90 = np.percentile(traj, 90, axis=0)
            fig_spend_median.add_trace(go.Scatter(x=years[2:], y=median[2:], stackgroup='one', fillcolor=colors[i % len(colors)], name=name))
            fig_spend_p10.add_trace(go.Scatter(x=years[2:], y=p10[2:], stackgroup='one', fillcolor=colors[i % len(colors)], name=name))
            fig_spend_p90.add_trace(go.Scatter(x=years[2:], y=p90[2:], stackgroup='one', fillcolor=colors[i % len(colors)], name=name))
        for f, title in zip([fig_spend_median, fig_spend_p10, fig_spend_p90],
                            ["Spending Median", "Spending 10% CL", "Spending 90% CL"]):
            f.update_layout(title=title, xaxis_title="Year", yaxis_title="Spending ($)", template="plotly_white", hovermode="x unified", height=400)

        # --- MAGI, Travel/Gifting, Roth, Medicare ---
        def simple_fig(name, traj, color="blue"):
            f = go.Figure()
            median = np.median(traj, axis=0)
            p10 = np.percentile(traj, 10, axis=0)
            p90 = np.percentile(traj, 90, axis=0)
            f.add_trace(go.Scatter(x=years[2:], y=median[2:], line=dict(color=color, width=4), name='Median'))
            f.add_trace(go.Scatter(x=years[2:], y=p10[2:], line=dict(color="red", width=2), name='10% CL'))
            f.add_trace(go.Scatter(x=years[2:], y=p90[2:], line=dict(color="green", width=2), name='90% CL'))
            f.update_layout(title=name, xaxis_title="Year", yaxis_title="Amount ($)", template="plotly_white", hovermode="x unified", height=400)
            return f

        fig_magi = simple_fig("MAGI", magi)
        fig_travel_gifting = simple_fig("Travel & Gifting", travel+gifting)
        fig_roth = simple_fig("ROTH Conversions", conversions)
        fig_medicare = simple_fig("Medicare Costs", medicare)

        # --- Return all outputs ---
        success_text = f"Success Rate: {success_rate:.2f}%    Avoid Ruin: {avoid_ruin_rate:.2f}%"
        success_style = {'fontSize': 30, 'color': vanguard_color(success_rate), 'textAlign': 'center'}
        sim_info = f"{nsims:,} simulations • {elapsed:.1f}s"

        debug_text = "\n".join(sim.debug_log)

        return (success_text, success_style, sim_info,
                fig_port_median, fig_port_p10, fig_port_p90,
                fig_income_median, fig_income_p10, fig_income_p90,
                fig_spend_median, fig_spend_p10, fig_spend_p90,
                fig_magi, fig_travel_gifting, fig_roth, fig_medicare,
                debug_text)

def vanguard_color(success_rate):
    sr = max(0, min(100, success_rate))
    if sr >= 75:
        r = int(255 * (1 - (sr - 75)/25)); g = 200; b = 0
    elif sr >= 50:
        r = 255; g = int(200 * ((sr - 50)/25 + 0.5)); b = 0
    else:
        r = 255; g = int(200 * (sr/50)); b = 0
    return f'rgb({r},{g},{b})'
