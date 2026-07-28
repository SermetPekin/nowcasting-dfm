"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

All plotting functions for the DFM package in one place, organised by concern:
  - Data transformation plots  (plot_transformed_data_*)
  - Model / convergence plots  (plot_loglik*, plot_common, plot_projection_*)
  - Factor analysis plots      (plot_factors_with_series, plot_prediction_intervals,
                                plot_factor_contribution)
  - Diagnostic plots           (plot_covariance_network, plot_em_step_deltas)
"""

from typing import Iterable
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from dfm_sp.sp_classes import Options, ResultObject
from dfm_sp.sp_utils import get_attr_from_spec


# ---------------------------------------------------------------------------
# Data transformation plots
# ---------------------------------------------------------------------------


def plot_transformed_data_one(Spec, X, Z, Time, series_id: Iterable = None, show=True):
    transformation_str = get_attr_from_spec(Spec, series_id, "Transformation")
    series_name = get_attr_from_spec(Spec, series_id, "SeriesName")
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
    fig.add_trace(
        go.Scatter(x=pd.to_datetime(Time[t_obs]).to_list(), y=Z[t_obs, idxSeries]),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=pd.to_datetime(Time[t_obs]).to_list(), y=X[t_obs, idxSeries]),
        row=2,
        col=1,
    )
    fig.update_layout(
        {"plot_bgcolor": "rgba(0, 0, 0, 0)"},
        title_text=f"[{series_id}] - {series_name}",
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
        fig.add_trace(
            go.Scatter(x=pd.to_datetime(Time[t_obs]).to_list(), y=Z[t_obs, idxSeries]),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=pd.to_datetime(Time[t_obs]).to_list(), y=X[t_obs, idxSeries]),
            row=2,
            col=1,
        )
        fig.update_layout(
            {"plot_bgcolor": "rgba(0, 0, 0, 0)"},
            title_text=f"[{series_id}] - {series_name}",
            showlegend=False,
        )
        fig.update_yaxes(title_text=Spec.Units[idxSeries], row=1, col=1)
        fig.update_yaxes(title_text=Spec.UnitsTransformed[idxSeries], row=2, col=1)
        if show:
            fig.show()
        result_plots.append(fig)
    return result_plots


# ---------------------------------------------------------------------------
# Model / convergence plots
# ---------------------------------------------------------------------------


def plot_loglik(res_obj: ResultObject, Time, show=True):
    """Plot log-likelihood convergence across EM iterations."""
    Res = res_obj.result
    options = res_obj.options
    assert isinstance(options, Options)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(Res["loglik"]) + 1),
            y=Res["loglik"],
            mode="lines",
            name="LogLik",
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        title_text=f"LogLik across number of steps (max iter: {options.max_iter})",
        showlegend=False,
    )
    fig.update_yaxes(title_text=f"LogLik - max iter: {options.max_iter}")
    fig.update_xaxes(title_text="Number of steps")
    if show:
        fig.show()
    return fig


def plot_common(res_obj: ResultObject, Time, series_id: str = "INDPRO", show=True):
    """Plot the global common factor overlaid on all standardised series."""
    Res = res_obj.result
    Spec = res_obj.spec
    options = res_obj.options
    assert isinstance(options, Options)
    if series_id not in Spec.SeriesID:
        raise ValueError(f"Series {series_id} not found in Spec.SeriesID")
    idxSeries = np.where(Spec.SeriesID == series_id)[0][0]
    dates = pd.to_datetime(Time).strftime("%Y-%m-%d").tolist()
    if Res["x_sm"].ndim != 2:
        raise ValueError("Res['x_sm'] must be a 2D array")
    fig = go.Figure()
    for i in range(Res["x_sm"].shape[1]):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=Res["x_sm"][:, i],
                mode="lines",
                name=Spec.SeriesID[i],
                line={"width": 0.9},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=Res["Z"][:, 0] * Res["C"][idxSeries, 0],
            mode="lines",
            name=f"Common Factor for {series_id}",
            line=dict(color="black", width=1.5),
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        title_text=f"Common Factor and Standardized Data for {series_id}",
    )
    if show:
        fig.show()
    return fig


def plot_loglik_together(res_obj: ResultObject, Time, show=True):
    """Return (fig_loglik, fig_common) in one call."""
    fig_loglik = plot_loglik(res_obj, Time, show=False)
    Res = res_obj.result
    Spec = res_obj.spec
    if "INDPRO" not in Spec.SeriesID:
        raise ValueError("Series 'INDPRO' not found in Spec.SeriesID")
    idxSeries = np.where(Spec.SeriesID == "INDPRO")[0][0]
    dates = pd.to_datetime(Time).strftime("%Y-%m-%d").tolist()
    if Res["x_sm"].ndim != 2:
        raise ValueError("Res['x_sm'] must be a 2D array")
    fig_common = go.Figure()
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
    if show:
        fig_loglik.show()
        fig_common.show()
    return fig_loglik, fig_common


def plot_projection_x_over_y(res_obj, X, Z, Time, series=None, show=True):
    """Plot projection of the common factor onto selected series."""
    Res = res_obj.result
    Spec = res_obj.spec
    options = res_obj.options
    assert isinstance(options, Options)
    if series is None:
        series = ["EMP", "GDP"]
    for s in series:
        if s not in Spec.SeriesID:
            raise ValueError(f"Series '{s}' not found in Spec.SeriesID")
    SeriesNames = [
        name_
        for s in series
        for id_, name_ in zip(Spec.SeriesID, Spec.SeriesName)
        if id_ == s
    ]
    dates = pd.to_datetime(Time).strftime("%Y-%m-%d").tolist()
    t_obs_dict = {
        s: ~np.isnan(X[:, np.where(Spec.SeriesID == s)[0][0]]) for s in series
    }
    fig = make_subplots(rows=2, cols=1, subplot_titles=SeriesNames)
    for i, s in enumerate(series):
        idxSeries = np.where(Spec.SeriesID == s)[0][0]
        t_obs = t_obs_dict[s]
        dates_obs = pd.to_datetime(Time[t_obs]).strftime("%Y-%m-%d").tolist()
        C_slice = Res["C"][idxSeries, :5]
        Z_slice = Res["Z"][:, :5]
        CommonFactor = (C_slice @ Z_slice.T) * Res["Wx"][idxSeries] + Res["Mx"][
            idxSeries
        ]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=CommonFactor,
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
        fig.update_yaxes(
            title_text=f"{Spec.Units[idxSeries]} ({Spec.UnitsTransformed[idxSeries]})",
            row=i + 1,
            col=1,
        )
    fig.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        title_text="Projection of Common Factor",
        height=600,
        showlegend=True,
    )
    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Factor analysis plots
# ---------------------------------------------------------------------------


def plot_factors_with_series(
    res_obj: ResultObject, Time, series_list: list, show: bool = True
) -> go.Figure:
    """Plot factors alongside selected series to interpret them."""
    Res = res_obj.result
    Spec = res_obj.spec
    Z = Res["Z"]
    x_sm = Res["x_sm"]
    dates = pd.to_datetime(Time).strftime("%Y-%m-%d").tolist()
    fig = go.Figure()
    for i in range(Z.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=Z[:, i],
                name=f"Factor {i+1}",
                line=dict(width=2),
            )
        )
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
    fig.update_layout(title="Factors vs. Selected Series", xaxis_title="Date")
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
    """Plot nowcasts with prediction intervals for a target series."""
    Res = res_obj.result
    Spec = res_obj.spec
    if target_series not in Spec.SeriesID:
        raise ValueError(f"Series '{target_series}' not found in Spec.SeriesID")
    idx_series = np.where(Spec.SeriesID == target_series)[0][0]
    dates = pd.to_datetime(Time).strftime("%Y-%m-%d").tolist()
    Z = Res["Z"]
    C = Res["C"]
    R = Res["R"]
    Q = Res["Q"]
    nowcast = Z @ C[idx_series, :].T
    var_nowcast = C[idx_series, :] @ Q @ C[idx_series, :].T + R[idx_series, idx_series]
    se_nowcast = np.sqrt(var_nowcast)
    z_score = norm.ppf(1 - (1 - confidence_level) / 2)
    lower_bound = nowcast - z_score * se_nowcast
    upper_bound = nowcast + z_score * se_nowcast
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=nowcast,
            mode="lines",
            name=f"Nowcast ({target_series})",
            line=dict(color="blue", width=2),
        )
    )
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
    C = Res["C"]
    R = Res["R"]
    Q = Res["Q"]
    total_var = np.diag(C @ Q @ C.T + R)
    factor_var = np.array(
        [[C[i, j] ** 2 * Q[j, j] for j in range(C.shape[1])] for i in range(C.shape[0])]
    )
    factor_pct = (factor_var.T / total_var).T * 100
    fig = go.Figure()
    for j in range(factor_pct.shape[1]):
        fig.add_trace(go.Bar(name=f"Factor {j+1}", x=Spec.SeriesID, y=factor_pct[:, j]))
    fig.update_layout(
        title="% of Variance Explained by Each Factor (per Series)",
        xaxis_title="Series",
        yaxis_title="% Variance",
        barmode="stack",
    )
    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------


def plot_covariance_network(res_obj: ResultObject, show: bool = True) -> go.Figure:
    """Render a cross-correlation heatmap of the Kalman-smoothed series."""
    Res = res_obj.result
    Spec = res_obj.spec
    x_sm = Res["x_sm"]
    series_names = Spec.SeriesID
    df_smoothed = pd.DataFrame(x_sm, columns=series_names)
    Corr = df_smoothed.corr().to_numpy(copy=True)
    np.fill_diagonal(Corr, 0)
    fig = go.Figure(
        data=go.Heatmap(
            z=Corr,
            x=series_names,
            y=series_names,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            zmid=0,
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title="Denoised Macroeconomic Correlation Network",
        xaxis_title="Macroeconomic Series",
        yaxis_title="Macroeconomic Series",
        width=900,
        height=800,
        xaxis={"tickangle": 45},
    )
    if show:
        fig.show()
    return fig


def plot_em_step_deltas(res_obj: ResultObject, show: bool = True) -> go.Figure:
    """Plot the per-iteration % change in log-likelihood to visualise EM convergence."""
    ll = np.array(res_obj.result["loglik"])
    with np.errstate(divide="ignore", invalid="ignore"):
        deltas = np.where(
            ll[:-1] != 0,
            np.diff(ll) / np.abs(ll[:-1]) * 100,
            0.0,
        )
    fig = go.Figure(
        go.Scatter(
            x=np.arange(2, len(ll) + 1),
            y=deltas,
            mode="lines+markers",
            name="Δ Log-Likelihood",
            marker=dict(size=6, color="red"),
        )
    )
    fig.update_layout(
        title="EM Algorithm Convergence Rate (% Change per Iteration)",
        xaxis_title="EM Iteration Step",
        yaxis_title="% Change Delta",
        template="plotly_white",
        yaxis=dict(type="log"),
    )
    if show:
        fig.show()
    return fig
