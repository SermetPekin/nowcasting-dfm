import numpy as np
import plotly.graph_objects as go
import pandas as pd
from typing import Optional
from dfm_sp.sp_classes import ResultObject


def plot_covariance_network(res_obj: ResultObject, show: bool = True) -> go.Figure:
    """
    Renders a normalized cross-correlation heatmap computed directly from the
    estimated/smoothed Factor outputs against the real transformed data.
    This shows which real variables inherently move together over time after
    being denoised by the Kalman Filter!
    """
    Res = res_obj.result
    Spec = res_obj.spec

    # x_sm contains the fully smoothed estimated macroeconomic data series (Numba EM processed)
    x_sm = Res["x_sm"]
    series_names = Spec.SeriesID

    # Compute classic Pearson Correlation Matrix
    # Using Pandas corr() on the smoothed Numba output paths to get perfectly clean correlations
    df_smoothed = pd.DataFrame(x_sm, columns=series_names)
    Corr = df_smoothed.corr().to_numpy(copy=True)

    # Zero out the diagonal (which is perfectly 1.0) so the color scale
    # focuses fully on highlighting cross-series structural relationships
    np.fill_diagonal(Corr, 0)

    fig = go.Figure(
        data=go.Heatmap(
            z=Corr,
            x=series_names,
            y=series_names,
            colorscale="RdBu",  # Red-Blue is the gold standard for -1 to 1 correlations
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
    """
    Plots the derivative of the Log-Likelihood curve (the step-by-step % change delta)
    to visually demonstrate the convergence efficiency of the EM algorithm.
    """
    Res = res_obj.result
    ll = np.array(Res["loglik"])

    # Calculate % drop per iteration
    deltas = np.diff(ll) / np.abs(ll[:-1]) * 100

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
        yaxis=dict(type="log"),  # Log scale helps see the asymptote
    )

    if show:
        fig.show()
    return fig
