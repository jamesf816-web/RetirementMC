# utils/plotting.py 

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import html
from models import PlannerInputs
import callbacks.simulation_callbacks as sim

# ------------------------------------------------------------------
# HELPER: Stacked area charts (Portfolio, Income, Spending)
# ------------------------------------------------------------------
def create_stacked_figure(trajectories, percentile, title, yaxis_title, color_theme="Vivid"):
    """
    trajectories = list of (label, data_array) where data_array is (n_sims, n_years)
    """
    fig = go.Figure()

    # Compute percentile for each category
    percentile_data = {
        label: np.percentile(data, percentile, axis=0)
        for label, data in trajectories
    }

    # Color palette
    colors = px.colors.qualitative.Vivid if color_theme == "Vivid" else px.colors.qualitative.Plotly

    for idx, (label, _) in enumerate(trajectories):
        y = percentile_data[label][2:]  # skip first 2 years

        fig.add_trace(go.Scatter(
            x=sim.years[2:], #skip first 2 years
            y=y,
            mode='lines',
            line=dict(width=0),
            fillcolor=colors[idx % len(colors)],
            stackgroup='one',
            name=label,
            hovertemplate=f'<b>{label}</b><br>Year: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
        ))

    percentile_suffix = " (Median)" if percentile == 50 else f" ({percentile}th percentile)"
    fig.update_layout(
        title=title + percentile_suffix,
        xaxis_title="Year",
        yaxis_title=yaxis_title,
        template="plotly_white",
        hovermode="x unified",
        height=500,
        legend=dict(x=1, y=1, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.9)")
    )
    return fig


# ------------------------------------------------------------------
# HELPER: Regular multi-line charts (MAGI, Travel/Gifting, etc.)
# ------------------------------------------------------------------
def create_multi_line_plot(trajectories_dict, title, yaxis_title):
    """
    trajectories_dict = {label: data_array}  # data_array shape (n_sims, n_years)
    """
    fig = go.Figure()

    for label, data in trajectories_dict.items():
        y = np.median(data, axis=0)[2:]  # median + skip first 2 years

        fig.add_trace(go.Scatter(
            x=sim.years[2:], #skip first 2 years
            y=y,
            mode='lines',
            name=label,
            line=dict(width=3),
            hovertemplate='Year: %{{x}}<br>{}: $%{{y:,.0f}}<extra></extra>'.format(label)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=yaxis_title,
        template="plotly_white",
        hovermode="x unified",
        height=400,
        legend=dict(x=1, y=1, xanchor="right", yanchor="top")
    )
    return fig

# ------------------------------------------------------------------
# HTML headers/tables
# ------------------------------------------------------------------
def create_rate_header(success_rate, success_style, ruin_avoidance):
    return html.Div([
        html.H3(f"Success Rate: {success_rate:.1f}%", style=success_style),
        html.H3(f"Ruin Avoidance: {ruin_avoidance:.1f}%", style=success_style),
    ], style={'display': 'flex', 'justifyContent': 'center', 'padding': '15px'})

def create_metrics_table(results, inputs, elapsed):
    final_median = results.get('final_median', np.nan)
    return html.Table([
        html.Tr([html.Th("Metric", style={'textAlign': 'left'}), html.Th("Value")]),
        html.Tr([html.Td("Simulation Time"), html.Td(f"{elapsed:.2f} s")]),
        html.Tr([html.Td("Final Median Portfolio"), html.Td(f"${final_median:,.0f}" if not np.isnan(final_median) else "N/A")]),
    ], style={'width': '100%', 'marginTop': '10px', 'borderCollapse': 'collapse'})


# ------------------------------------------------------------------
# MAIN FUNCTION 
# ------------------------------------------------------------------
def generate_all_plots(results: dict, inputs, elapsed):
    # Extract paths
    portfolio = results["portfolio_paths"]
    travel = results.get("travel_paths", np.zeros_like(portfolio))
    gifting = results.get("gifting_paths", np.zeros_like(portfolio))
    base_spending = results.get("base_spending_paths", np.zeros_like(portfolio))
    lumpy_spending = results.get("lumpy_spending_paths", np.zeros_like(portfolio))

    rmds = results.get("rmd_paths", np.zeros_like(portfolio))
    ssbenefit = results.get("ssbenefit_paths", np.zeros_like(portfolio))
    portfolio_withdrawal = results.get("portfolio_withdrawal_paths", np.zeros_like(portfolio))
    def457b_income = results.get("def457b_income_paths", np.zeros_like(portfolio))
    pension = results.get("pension_paths", np.zeros_like(portfolio))
    trust_income = results.get("trust_income_paths", np.zeros_like(portfolio))

    taxes = results.get("taxes_paths", np.zeros_like(portfolio))
    conversion = results.get("conversion_paths", np.zeros_like(portfolio))
    medicare = results.get("medicare_paths", np.zeros_like(portfolio))
    magi = results.get("magi_paths", np.zeros_like(portfolio))

    # --------------------------------------------------
    # 1. Portfolio by tax type
    # --------------------------------------------------
    tax_type_map = {
        "def457b": "457b Deferred Salary",
        "taxable": "Taxable Brokerage",
        "trust": "Trust Accounts",
        "traditional": "Traditional IRAs",
        "inherited": "Inherited IRAs",
        "roth": "Roth IRAs"
    }

    portfolio_trajectories = []
    n_sims, n_years = results["portfolio_paths"].shape

    for tax, label in tax_type_map.items():
        paths = np.zeros((n_sims, n_years))
        for name, traj in results["account_paths"].items():
            if inputs.accounts[name]["tax"] == tax:
                paths += traj
        if paths.max() > 0:
            portfolio_trajectories.append((label, paths))

    # --------------------------------------------------
    # 2. Income & Spending trajectories
    # --------------------------------------------------
    income_trajectories = [
        ("457b Deferred Salary", def457b_income),
        ("Trust Income", trust_income),
        ("RMDs", rmds),
        ("Pension", pension),
        ("SS Benefit", ssbenefit),
        ("Portfolio Withdrawal", portfolio_withdrawal),
    ]

    spending_trajectories = [
        ("Base Spending", base_spending),
        ("Travel", travel),
        ("Gifting", gifting),
        ("Taxes", taxes),
        ("Lumpy Spending", lumpy_spending),
    ]

    # --------------------------------------------------
    # Generate all figures
    # --------------------------------------------------
    all_figures = {}

    # Stacked charts
    all_figures["portfolio-median"] = create_stacked_figure(portfolio_trajectories, 50, "Portfolio by Tax Type", "Portfolio Balance ($)", "Vivid")
    all_figures["portfolio-p10"]    = create_stacked_figure(portfolio_trajectories, 10, "Portfolio by Tax Type", "Portfolio Balance ($)", "Vivid")
    all_figures["portfolio-p90"]    = create_stacked_figure(portfolio_trajectories, 90, "Portfolio by Tax Type", "Portfolio Balance ($)", "Vivid")

    all_figures["income-median"] = create_stacked_figure(income_trajectories, 50, "Income Sources", "Annual Income ($)", "Vivid")
    all_figures["income-p10"]    = create_stacked_figure(income_trajectories, 10, "Income Sources", "Annual Income ($)", "Vivid")
    all_figures["income-p90"]    = create_stacked_figure(income_trajectories, 90, "Income Sources", "Annual Income ($)", "Vivid")

    all_figures["spending-median"] = create_stacked_figure(spending_trajectories, 50, "Spending Categories", "Annual Spending ($)", "Vivid")
    all_figures["spending-p10"]    = create_stacked_figure(spending_trajectories, 10, "Spending Categories", "Annual Spending ($)", "Vivid")
    all_figures["spending-p90"]    = create_stacked_figure(spending_trajectories, 90, "Spending Categories", "Annual Spending ($)", "Vivid")

    # Line charts
    all_figures["magi"]            = create_multi_line_plot({"MAGI": magi}, "Modified Adjusted Gross Income (MAGI)", "MAGI ($)")
    all_figures["travel-gifting"]  = create_multi_line_plot({"Travel": travel, "Gifting": gifting}, "Travel & Gifting Spending", "Annual Amount ($)")
    all_figures["roth-conversions"] = create_multi_line_plot({"Roth Conversions": conversion}, "Roth Conversions", "Annual Conversion ($)")
    all_figures["medicare-costs"]  = create_multi_line_plot({"Medicare/IRMA Costs": medicare}, "Medicare Premiums (incl. IRMAA)", "Annual Cost ($)")

    # Headers — ALWAYS compute fresh, never trust stale data_dict value
    success_rate = results.get("success_rate", 0.0)
    ruin_rate = results.get("avoid_ruin_rate", 0.0)
    
    color = vanguard_color(success_rate)
    success_style = {
        "color": color,
        "fontWeight": "bold",
        "fontSize": "28px",
        "margin": "0 40px",
        "textAlign": "center"
    }
    
    rate_header = create_rate_header(success_rate, success_style, ruin_rate)
    metrics_table = create_metrics_table(results, inputs, elapsed)

    return all_figures, rate_header, metrics_table


def get_figure_ids():
    return [
        "portfolio-median", "income-median", "spending-median",
        "magi", "travel-gifting", "roth-conversions", "medicare-costs",
        "portfolio-p10", "income-p10", "spending-p10",
        "portfolio-p90", "income-p90", "spending-p90"
    ]

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
