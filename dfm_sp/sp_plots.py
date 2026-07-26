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


def plot_transformed_data_one(Spec, X, Z, Time, series_id: Iterable = None, show=True):
    transformation_str = get_attr_from_spec(Spec, series_id, "Transformation")
    series_name = get_attr_from_spec(Spec, series_id, "SeriesName")
    # -------------------------------------------------Plot data
    # Raw vs transformed
    idxSeries = np.where(Spec.SeriesID == series_id)[0][0]
    t_obs = ~np.isnan(X[:, idxSeries])
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"Raw Observed Data [{series_id}]",
            f"Transformed Data [{series_id}] => [{transformation_str}]",
        ),
    )
    fig.append_trace(
        go.Scatter(
            # (Legacy ordinal comment removed)
            x=pd.to_datetime(Time[t_obs]).to_list(),
            y=Z[t_obs, idxSeries],
        ),
        row=1,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            # (Legacy ordinal comment removed)
            x=pd.to_datetime(Time[t_obs]).to_list(),
            y=X[t_obs, idxSeries],
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        {"plot_bgcolor": "rgba(0, 0, 0, 0)"},
        title_text=f"[{series_id }] - {series_name}",
        showlegend=False,
    )
    fig.update_yaxes(title_text=Spec.Units[idxSeries], row=1, col=1)
    fig.update_yaxes(title_text=Spec.UnitsTransformed[idxSeries], row=2, col=1)
    if show:
        fig.show()
    return fig


def plot_transformed_data(Spec, X, Z, Time, series_wanted: Iterable = None, show=True):
    if series_wanted is None:
        series_wanted = ["INDPRO", "RSA", "PPI", "CIRC"]
    result_plots = []
    for series_id in series_wanted:
        transformation_str = get_attr_from_spec(Spec, series_id, "Transformation")
        series_name = get_attr_from_spec(Spec, series_id, "SeriesName")
        # -------------------------------------------------Plot data
        # Raw vs transformed
        idxSeries = np.where(Spec.SeriesID == series_id)[0][0]
        t_obs = ~np.isnan(X[:, idxSeries])
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                f"Raw Observed Data [{series_id}]",
                f"Transformed Data [{series_id}] => [{transformation_str}]",
            ),
        )
        fig.append_trace(
            go.Scatter(
                # (Legacy ordinal comment removed)
                x=pd.to_datetime(Time[t_obs]).to_list(),
                y=Z[t_obs, idxSeries],
            ),
            row=1,
            col=1,
        )
        fig.append_trace(
            go.Scatter(
                # (Legacy ordinal comment removed)
                x=pd.to_datetime(Time[t_obs]).to_list(),
                y=X[t_obs, idxSeries],
            ),
            row=2,
            col=1,
        )
        fig.update_layout(
            {"plot_bgcolor": "rgba(0, 0, 0, 0)"},
            title_text=f"[{series_id }] - {series_name}",
            showlegend=False,
        )
        fig.update_yaxes(title_text=Spec.Units[idxSeries], row=1, col=1)
        fig.update_yaxes(title_text=Spec.UnitsTransformed[idxSeries], row=2, col=1)
        if show:
            fig.show()
        result_plots.append(fig)
    return result_plots


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_loglik(res_obj: ResultObject, Time, show=True):
    """Plot log-likelihood and common factor vs. standardized data.
    Args:
        res_obj: ResultObject containing results and specifications.
        Time: Array of datetime64 values (n_time,).
        show: If True, display the figures (default: True).
    Returns:
        fig_loglik: Figure for log-likelihood plot.
        fig_common: Figure for common factor plot.
    """
    Time
    # Extract results and specs
    Res = res_obj.result
    Spec = res_obj.spec
    options = res_obj.options
    assert isinstance(options, Options), "options must be a RunOptions instance"
    # --- Plot 1: Log-likelihood ---
    fig_loglik = go.Figure()
    fig_loglik.add_trace(
        go.Scatter(
            x=np.arange(1, len(Res["loglik"]) + 1),  # Include all steps
            y=Res["loglik"],
            mode="lines",
            name="LogLik",
        )
    )
    fig_loglik.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        title_text=f"LogLik across number of steps (max iter: {options.max_iter})",
        showlegend=False,
    )
    fig_loglik.update_yaxes(title_text=f"LogLik - max iter: {options.max_iter}")
    fig_loglik.update_xaxes(title_text="Number of steps")
    if show:
        fig_loglik.show()
    return fig_loglik


def plot_common(res_obj: ResultObject, Time, series_id: str = "INDPRO", show=True):
    """Plot log-likelihood and common factor vs. standardized data.
    Args:
        res_obj: ResultObject containing results and specifications.
        Time: Array of datetime64 values (n_time,).
        show: If True, display the figures (default: True).
    Returns:
        fig_common: Figure for common factor plot.
    """
    # Extract results and specs
    Res = res_obj.result
    Spec = res_obj.spec
    options = res_obj.options
    assert isinstance(options, Options), "options must be a RunOptions instance"
    # --- Plot 2: Common factor and standardized data ---
    # Validate series_id  exists
    if series_id not in Spec.SeriesID:
        raise ValueError(f"Series {series_id} not found in Spec.SeriesID")
    idxSeries = np.where(Spec.SeriesID == series_id)[0][0]
    # Precompute dates
    dates = pd.to_datetime(Time).strftime("%Y-%m-%d").tolist()
    # Check x_sm is 2D
    if Res["x_sm"].ndim != 2:
        raise ValueError("Res['x_sm'] must be a 2D array")
    fig_common = go.Figure()
    # Plot standardized data
    for i in range(Res["x_sm"].shape[1]):
        fig_common.add_trace(
            go.Scatter(
                x=dates,
                y=Res["x_sm"][:, i],
                mode="lines",
                name=Spec.SeriesID[i],
                line={"width": 0.9},
            )
        )
    # Plot common factor
    fig_common.add_trace(
        go.Scatter(
            x=dates,
            y=Res["Z"][:, 0] * Res["C"][idxSeries, 0],
            mode="lines",
            name=f"Common Factor for {series_id}",
            line=dict(color="black", width=1.5),
        )
    )
    fig_common.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        title_text=f"Common Factor and Standardized Data for {series_id}",
    )
    if show:
        fig_common.show()
    return fig_common


def plot_loglik_together(res_obj: ResultObject, Time, show=True):
    """Plot log-likelihood and common factor vs. standardized data.
    Args:
        res_obj: ResultObject containing results and specifications.
        Time: Array of datetime64 values (n_time,).
        show: If True, display the figures (default: True).
    Returns:
        fig_loglik: Figure for log-likelihood plot.
        fig_common: Figure for common factor plot.
    """
    # Extract results and specs
    Res = res_obj.result
    Spec = res_obj.spec
    options = res_obj.options
    assert isinstance(options, Options), "options must be a RunOptions instance"
    # --- Plot 1: Log-likelihood ---
    fig_loglik = go.Figure()
    fig_loglik.add_trace(
        go.Scatter(
            x=np.arange(1, len(Res["loglik"]) + 1),  # Include all steps
            y=Res["loglik"],
            mode="lines",
            name="LogLik",
        )
    )
    fig_loglik.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        title_text=f"LogLik across number of steps (max iter: {options.max_iter})",
        showlegend=False,
    )
    fig_loglik.update_yaxes(title_text=f"LogLik - max iter: {options.max_iter}")
    fig_loglik.update_xaxes(title_text="Number of steps")
    # --- Plot 2: Common factor and standardized data ---
    # Validate INDPRO exists
    if "INDPRO" not in Spec.SeriesID:
        raise ValueError("Series 'INDPRO' not found in Spec.SeriesID")
    idxSeries = np.where(Spec.SeriesID == "INDPRO")[0][0]
    # Precompute dates
    dates = pd.to_datetime(Time).strftime("%Y-%m-%d").tolist()
    # Check x_sm is 2D
    if Res["x_sm"].ndim != 2:
        raise ValueError("Res['x_sm'] must be a 2D array")
    fig_common = go.Figure()
    # Plot standardized data
    for i in range(Res["x_sm"].shape[1]):
        fig_common.add_trace(
            go.Scatter(
                x=dates,
                y=Res["x_sm"][:, i],
                mode="lines",
                name=Spec.SeriesID[i],
                line={"width": 0.9},
            )
        )
    # Plot common factor
    fig_common.add_trace(
        go.Scatter(
            x=dates,
            y=Res["Z"][:, 0] * Res["C"][idxSeries, 0],
            mode="lines",
            name="Common Factor",
            line=dict(color="black", width=1.5),
        )
    )
    fig_common.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        title_text="Common Factor and Standardized Data",
    )
    # Show figures if requested
    if show:
        fig_loglik.show()
        fig_common.show()
    return fig_loglik, fig_common


def plot_projection_x_over_y(res_obj, X, Z, Time, series=None, show=True):
    """Plot projection of common factor onto Payroll Employment and GDP.
    Args:
        res_obj: ResObject containing results and specifications.
        X: 2D array of observed data (n_time x n_series).
        Z: 2D array of latent factors (n_time x n_factors).
        Time: Array of datetime64 values (n_time,).
    Returns:
        fig: Plotly figure object with two subplots (Employment and GDP).
    """
    # Extract results and specs
    Res = res_obj.result
    Spec = res_obj.spec
    options = res_obj.options
    assert isinstance(options, Options), "options must be a RunOptions instance"
    if series is None:
        series = ["EMP", "GDP"]
    # Validate series exist in Spec.SeriesID
    for s in series:
        if s not in Spec.SeriesID:
            raise ValueError(f"Series '{s}' not found in Spec.SeriesID")
    SeriesNames = []
    for s in series:
        for id_, name_ in zip(Spec.SeriesID, Spec.SeriesName):
            if id_ == s:
                SeriesNames.append(name_)
    # Precompute all dates once
    dates = pd.to_datetime(Time).strftime("%Y-%m-%d").tolist()
    # Precompute observation masks for all series
    t_obs_dict = {
        s: ~np.isnan(X[:, np.where(Spec.SeriesID == s)[0][0]]) for s in series
    }
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=SeriesNames,  # series
        # subplot_titles=("Employment", "Gross Domestic Product")
    )
    # Plot each series
    for i, s in enumerate(series):
        idxSeries = np.where(Spec.SeriesID == s)[0][0]
        t_obs = t_obs_dict[s]
        dates_obs = pd.to_datetime(Time[t_obs]).strftime("%Y-%m-%d").tolist()
        # Compute common factor
        C_slice = Res["C"][idxSeries, :5]  # Shape: (5,)
        Z_slice = Res["Z"][:, :5]  # Shape: (n_time, 5)
        CommonFactor = (C_slice @ Z_slice.T) * Res["Wx"][idxSeries] + Res["Mx"][
            idxSeries
        ]
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=CommonFactor,  #  [0, :],
                name=f"Common Factor ({s})",
                line=dict(color="blue"),
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates_obs,
                y=X[t_obs, idxSeries],
                name=f"Data ({s})",
                line=dict(color="red"),
            ),
            row=i + 1,
            col=1,
        )
        # Update y-axis label
        fig.update_yaxes(
            title_text=f"{Spec.Units[idxSeries]} ({Spec.UnitsTransformed[idxSeries]})",
            row=i + 1,
            col=1,
        )
    # Update layout
    fig.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        title_text="Projection of Common Factor",
        height=600,
        showlegend=True,
    )
    if show:
        fig.show()
    return fig
