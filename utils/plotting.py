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
def create_stacked_figure(trajectories, percentile, title, yaxis_title, color_theme, start_index, y_max=None):
    """
    Super-robust version – handles None, empty lists, zero-sim arrays, etc.
    """
    fig = go.Figure()

    # ------------------------------------------------------------
    # 1. Total guard – no trajectories at all
    # ------------------------------------------------------------
    if not trajectories or len(trajectories) == 0:
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=20
        )
        fig.update_layout(title=title, height=500, template="plotly_white")
        return fig

    # ------------------------------------------------------------
    # 2. Guard – any trajectory is None or has zero simulations
    # ------------------------------------------------------------
    valid_trajectories = []
    for label, data in trajectories:
        # data can be None, np.array, or list → convert to np.array safely
        if data is None:
            continue
        arr = np.asarray(data)
        if arr.size == 0 or arr.shape[0] == 0:
            continue  # skip completely empty series
        valid_trajectories.append((label, arr))

    if not valid_trajectories:
        fig.add_annotation(
            text="0 successful simulations",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=20
        )
        fig.update_layout(title=title, height=500, template="plotly_white")
        return fig

    # ------------------------------------------------------------
    # 3. Now we actually have real data → safe to compute percentiles
    # ------------------------------------------------------------
    percentile_data = {
        label: np.percentile(arr, percentile, axis=0)
        for label, arr in valid_trajectories
    }

    colors = px.colors.qualitative.Vivid if color_theme == "Vivid" else px.colors.qualitative.Plotly
    start_index = 2 # defauklt to skip 2 prior year
    for idx, (label, _) in enumerate(valid_trajectories):
        y = percentile_data[label][start_index:]  # skip to start_index

        fig.add_trace(go.Scatter(
            x=sim.years[start_index:], # skip to start_index
            y=y,
            mode='lines',
            line=dict(width=0),
            fillcolor=colors[idx % len(colors)],
            stackgroup='one',
            name=label,
            hovertemplate=f'<b>{label}</b><br>Year: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
        ))

    suffix = " (Median)" if percentile == 50 else f" ({percentile}th percentile)"
    fig.update_layout(
        title=title + suffix,
        xaxis_title="Year",
        yaxis_title=yaxis_title,
        yaxis=dict(range=[0, y_max]),
        template="plotly_white",
        hovermode="x unified",
        height=500,
        legend=dict(x=1, y=1, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.9)")
    )
    return fig


# ------------------------------------------------------------------
# HELPER: Regular multi-line charts (MAGI, Travel/Gifting, etc.)
def create_multi_line_plot(trajectories_dict, title, yaxis_title, start_index, y_max=None):
    """
    Plots median + 10th/90th percentiles.
    Now 100% safe against None, empty arrays, or zero simulations.
    """
    fig = go.Figure()

    # --------------------------------------------------
    # Guard: no data at all
    # --------------------------------------------------
    if not trajectories_dict or all(v is None for v in trajectories_dict.values()):
        fig.add_annotation(
            text="No data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font_size=20
        )
        fig.update_layout(title=title, height=400, template="plotly_white")
        return fig

    # --------------------------------------------------
    # Convert everything to numpy arrays safely
    # --------------------------------------------------
    valid_data = {}
    for label, data in trajectories_dict.items():
        if data is None:
            continue
        arr = np.asarray(data)
        if arr.size == 0 or arr.shape[0] == 0:
            continue
        valid_data[label] = arr

    if not valid_data:
        fig.add_annotation(
            text="0 successful simulations",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font_size=20
        )
        fig.update_layout(title=title, height=400, template="plotly_white")
        return fig

    # --------------------------------------------------
    # We have real data → plot it
    # --------------------------------------------------
    start_index = 2 # defauklt to skip 2 prior year
    for label, data in valid_data.items():
        data_trimmed = data[:, start_index:] if data.shape[1] > 2 else data #this slices to start at start_index
        years = sim.years[start_index:] if len(sim.years) > 2 else sim.years #this slices to start at start_index

        y_median = np.median(data_trimmed, axis=0)
        fig.add_trace(go.Scatter(
            x=years,
            y=y_median,
            mode='lines',
            name=label,
            line=dict(width=3),
            hovertemplate=f'<b>{label}</b><br>Year: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
        ))

        p10 = np.percentile(data_trimmed, 10, axis=0)
        p90 = np.percentile(data_trimmed, 90, axis=0)
        color = fig.data[-1].line.color

        fig.add_trace(go.Scatter(x=years, y=p90, line=dict(color=color, width=1.5, dash='dash'),
                                 name=f"{label} 90th", showlegend=False))
        fig.add_trace(go.Scatter(x=years, y=p10, line=dict(color=color, width=1.5, dash='dash'),
                                 name=f"{label} 10th", showlegend=False))

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=yaxis_title,
        yaxis=dict(range=[0, y_max]),
        template="plotly_white",
        hovermode="x unified",
        height=400,
        legend=dict(x=1, y=1, xanchor="right", yanchor="top")
    )
    return fig

# ------------------------------------------------------------------
# MAIN FUNCTION 
# ------------------------------------------------------------------
def generate_all_plots(results: dict, inputs, elapsed):
    # Extract paths
    portfolio = results["portfolio_paths"]
    travel = results.get("travel_paths", np.zeros_like(portfolio))
    gifting = results.get("gifting_paths", np.zeros_like(portfolio))
    base_spending = results.get("base_spending_paths", np.zeros_like(portfolio))
    mortgage_expense = results.get("mortgage_expense_paths", np.zeros_like(portfolio))
    lumpy_spending = results.get("lumpy_spending_paths", np.zeros_like(portfolio))

    rmds = results.get("rmd_paths", np.zeros_like(portfolio))
    ssbenefit = results.get("ssbenefit_paths", np.zeros_like(portfolio))
    portfolio_withdrawal = results.get("portfolio_withdrawal_paths", np.zeros_like(portfolio))
    salary = results.get("salary_paths", np.zeros_like(portfolio))
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
        "inherited": "Inherited IRAs",
        "traditional": "Traditional IRAs",
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
        ("Salary", salary),
        ("457b Deferred Salary", def457b_income),
        ("Pension", pension),
        ("SS Benefit", ssbenefit),
        ("Trust Income", trust_income),
        ("RMDs", rmds),
        ("Portfolio Withdrawal", portfolio_withdrawal),
    ]

    spending_trajectories = [
        ("Base Spending", base_spending),
        ("Mortgage Expense, Tax, Ins.", mortgage_expense),
        ("Taxes", taxes),
        ("Travel", travel),
        ("Gifting", gifting),
        ("Lumpy Spending", lumpy_spending),
    ]

    # --------------------------------------------------
    # Generate all figures
    # --------------------------------------------------
    all_figures = {}

    # Stacked median charts
    all_figures["portfolio-median"] = create_stacked_figure(portfolio_trajectories, 50, "Portfolio by Tax Type", "Portfolio Balance ($)", "Vivid", 1, None)
    all_figures["income-median"] = create_stacked_figure(income_trajectories, 50, "Income Sources", "Annual Income ($)", "Plotly", 2, 1000000)
    all_figures["spending-median"] = create_stacked_figure(spending_trajectories, 50, "Spending Categories", "Annual Spending ($)", "Plotly", 2, 1000000)

    # Line charts
    all_figures["magi"]            = create_multi_line_plot({"MAGI": magi}, "Modified Adjusted Gross Income (MAGI)", "MAGI ($)", 2, 1000000)
    all_figures["travel-gifting"]  = create_multi_line_plot({"Travel": travel, "Gifting": gifting}, "Travel & Gifting Spending", "Annual Amount ($)", 2, 500000)
    all_figures["roth-conversions"] = create_multi_line_plot({"Roth Conversions": conversion}, "Roth Conversions", "Annual Conversion ($)", 2, 300000)
    all_figures["medicare-costs"]  = create_multi_line_plot({"Medicare/IRMA Costs": medicare}, "Medicare Premiums (incl. IRMAA)", "Annual Cost ($)", 2)

    # Stacked 10% CL charts
    all_figures["portfolio-p10"]    = create_stacked_figure(portfolio_trajectories, 10, "Portfolio by Tax Type", "Portfolio Balance ($)", "Vivid", 1)
    all_figures["income-p10"]    = create_stacked_figure(income_trajectories, 10, "Income Sources", "Annual Income ($)", "Plotly", 2)
    all_figures["spending-p10"]    = create_stacked_figure(spending_trajectories, 10, "Spending Categories", "Annual Spending ($)", "Plotly", 2)

    # Stacked 90% CL charts
    all_figures["portfolio-p90"]    = create_stacked_figure(portfolio_trajectories, 90, "Portfolio by Tax Type", "Portfolio Balance ($)", "Vivid", 1)
    all_figures["income-p90"]    = create_stacked_figure(income_trajectories, 90, "Income Sources", "Annual Income ($)", "Plotly", 2)
    all_figures["spending-p90"]    = create_stacked_figure(spending_trajectories, 90, "Spending Categories", "Annual Spending ($)", "Plotly", 2)

    return [all_figures[id] for id in get_figure_ids()]


def get_figure_ids():
    return [
        "portfolio-median", "income-median", "spending-median",
        "magi", "travel-gifting", "roth-conversions", "medicare-costs",
        "portfolio-p10", "income-p10", "spending-p10",
        "portfolio-p90", "income-p90", "spending-p90"
    ]

