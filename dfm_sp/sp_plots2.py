import numpy as np
import plotly.graph_objects as go
import pandas as pd
from typing import Optional
from typing import Iterable
import numpy as np
from datetime import datetime as dt
from pathlib import Path

# -------------------------------------------------Libraries
# Libs
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dfm_sp.sp_classes import Options, ResultObject
from dfm_sp.sp_utils import get_attr_from_spec
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import norm


def plot_factors_with_series(
    res_obj: ResultObject, Time, series_list: list, show: bool = True
) -> go.Figure:
    """Plot factors alongside selected series to interpret them."""
    Res = res_obj.result
    Spec = res_obj.spec
    Z = Res["Z"]  # (n_time × n_factors)
    x_sm = Res["x_sm"]  # (n_time × n_series)
    dates = pd.to_datetime(Time).strftime("%Y-%m-%d").tolist()
    fig = go.Figure()
    # Plot factors
    for i in range(Z.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=Z[:, i],
                name=f"Factor {i+1}",
                line=dict(width=2),
            )
        )
    # Plot selected series
    for series in series_list:
        idx = np.where(Spec.SeriesID == series)[0][0]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=x_sm[:, idx],
                name=series,
                line=dict(dash="dot", width=1),
            )
        )
    fig.update_layout(
        title=f"Factors vs. Selected Series)",
        xaxis_title="Date",
    )
    if show:
        fig.show()
    return fig


def plot_prediction_intervals(
    res_obj: ResultObject,
    Time: np.ndarray,
    target_series: str = "INDPRO",
    confidence_level: float = 0.95,
    show: bool = True,
) -> go.Figure:
    """Plot nowcasts with prediction intervals for a target series.
    Args:
        res_obj: ResultObject containing results and specifications.
        Time: Array of datetime64 values (n_time,).
        target_series: Name of the series to plot (default: "INDPRO").
        confidence_level: Confidence level for intervals (e.g., 0.95 for 95% CI).
        show: If True, display the figure (default: True).
    Returns:
        fig: Plotly figure with nowcasts and prediction intervals.
    """
    Res = res_obj.result
    Spec = res_obj.spec
    # Validate target_series exists
    if target_series not in Spec.SeriesID:
        raise ValueError(f"Series '{target_series}' not found in Spec.SeriesID")
    idx_series = np.where(Spec.SeriesID == target_series)[0][0]
    # Precompute dates
    dates = pd.to_datetime(Time).strftime("%Y-%m-%d").tolist()
    # Extract model matrices
    Z = Res["Z"]  # Common factors (n_time × n_factors)
    C = Res["C"]  # Factor loadings (n_series × n_factors)
    R = Res["R"]  # Observation noise covariance (n_series × n_series)
    Q = Res["Q"]  # Factor noise covariance (n_factors × n_factors)
    # Nowcast: y = Z * C^T (assuming C is n_series × n_factors)
    nowcast = Z @ C[idx_series, :].T  # (n_time,) nowcast for target_series
    # Variance of nowcast:
    # Var(y) = C * Var(Z) * C^T + R
    # Here, Var(Z) is approximated by Q (covariance of factor noise)
    var_nowcast = C[idx_series, :] @ Q @ C[idx_series, :].T + R[idx_series, idx_series]
    # Standard error
    se_nowcast = np.sqrt(var_nowcast)
    # Critical value for confidence interval
    z_score = norm.ppf(1 - (1 - confidence_level) / 2)
    # Prediction intervals
    lower_bound = nowcast - z_score * se_nowcast
    upper_bound = nowcast + z_score * se_nowcast
    # Plot
    fig = go.Figure()
    # Nowcast line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=nowcast,
            mode="lines",
            name=f"Nowcast ({target_series})",
            line=dict(color="blue", width=2),
        )
    )
    # Confidence band
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=upper_bound,
            mode="lines",
            name=f"{int(confidence_level * 100)}% PI",
            line=dict(width=0),
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=lower_bound,
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(68, 138, 255, 0.2)",
            fill="tonexty",
            showlegend=False,
        )
    )
    # Add actual data if available (e.g., in x_sm or X_sm)
    if "x_sm" in Res:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=Res["x_sm"][:, idx_series],
                mode="lines",
                name=f"Actual ({target_series})",
                line=dict(color="black", width=1.5, dash="dot"),
            )
        )
    # Update layout
    fig.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        title_text=f"Nowcast with {int(confidence_level * 100)}% Prediction Intervals for {target_series}",
        xaxis_title="Date",
        yaxis_title=target_series,
    )
    if show:
        fig.show()
    return fig


def plot_factor_contribution(res_obj: ResultObject, show: bool = True) -> go.Figure:
    """Plot the % of variance in each series explained by each factor."""
    Res = res_obj.result
    Spec = res_obj.spec
    C = Res["C"]  # (n_series × n_factors)
    R = Res["R"]  # (n_series × n_series)
    # Total variance for each series: Var(y) = C @ Q @ C.T + R
    Q = Res["Q"]  # (n_factors × n_factors)
    total_var = np.diag(C @ Q @ C.T + R)  # (n_series,)
    # Variance explained by each factor: C[:, i]^2 * Q[i, i]
    factor_var = np.array(
        [[C[i, j] ** 2 * Q[j, j] for j in range(C.shape[1])] for i in range(C.shape[0])]
    )  # (n_series × n_factors)
    # Normalize to % of total variance
    factor_pct = (factor_var.T / total_var).T * 100
    fig = go.Figure()
    for j in range(factor_pct.shape[1]):
        fig.add_trace(
            go.Bar(
                name=f"Factor {j+1}",
                x=Spec.SeriesID,
                y=factor_pct[:, j],
            )
        )
    fig.update_layout(
        title="% of Variance Explained by Each Factor (per Series)",
        xaxis_title="Series",
        yaxis_title="% Variance",
        barmode="stack",
    )
    if show:
        fig.show()
    return fig
