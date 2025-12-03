# utils/plotting.py   ←  final working version (copy-paste this exactly)

import plotly.graph_objects as go
import numpy as np

def create_balance_paths_fig(account_paths, years, percentiles=(5, 25, 50, 75, 95), title="Portfolio Balance Paths"):
    """
    Expected signature used in the rest of the codebase
    account_paths: 2-D numpy array, shape (n_sims, n_years)
    years: 1-D array or list of years
    """
    fig = go.Figure()

    # Median (50th percentile)
    median = np.percentile(account_paths, 50, axis=0)
    fig.add_trace(go.Scatter(x=years, y=median, mode='lines', name='Median', line=dict(width=3, color='blue')))

    # Percentile bands
    colors = ['rgba(0,100,255,0.1)', 'rgba(0,100,255,0.25)']
    for i, (lower, upper) in enumerate([(5, 95), (25, 75)]):
        low = np.percentile(account_paths, lower, axis=0)
        high = np.percentile(account_paths, upper, axis=0)
        fig.add_trace(go.Scatter(x=years, y=high, fill=None, mode='lines',
                                 line=dict(color='rgba(255,255,255,0)'), showlegend=False))
        fig.add_trace(go.Scatter(x=years, y=low, fill='tonexty', mode='lines',
                                 fillcolor=colors[i], line=dict(color='rgba(255,255,255,0)'),
                                 name=f'{lower}th–{upper}th percentile'))

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Portfolio Balance ($)",
        template="simple_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


def create_ending_balance_hist(ending_balances, bins=60, title="Distribution of Ending Balances"):
    """
    Expected signature used in the rest of the codebase
    ending_balances: 1-D array of final portfolio values across simulations
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ending_balances,
        nbinsx=bins,
        name="Ending Balances",
        marker_color='royalblue',
        opacity=0.75
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Ending Balance After Retirement ($)",
        yaxis_title="Number of Simulations",
        template="simple_white",
        bargap=0.1
    )
    return fig


# Optional: provide dummy data so the layout can render before any simulation runs
if 'DUMMY_DATA' not in globals():
    YEARS = list(range(2025, 2066))
    DUMMY_PATHS = np.linspace(100_000, 3_000_000, 1000)[:, None] * np.ones((1, len(YEARS)))
    DUMMY_ENDING = np.random.lognormal(mean=14, sigma=0.8, size=5000)  # ~$1–$10M range
