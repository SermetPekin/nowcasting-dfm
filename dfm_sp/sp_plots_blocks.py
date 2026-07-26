"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dfm_sp.sp_classes import ResultObject


def plot_block_contributions(
    res_object: ResultObject, spec, time_array: np.ndarray, target_series: str
) -> go.Figure:
    """
    Generates a Stacked Bar Chart decomposing the Nowcast target into independent structural blocks
    (e.g., Global vs. Soft vs. Real).

    Args:
        res_object: The parsed DFMResult containing `result` (the Kalman smoother output).
        spec: The generic `SpecConfig` or `load_spec` containing the block matrices.
        target_series: The string series ID to trace (e.g. 'GDPC1')
    """
    # 1. Identify target series row in matrices
    series_idx = np.where(spec.SeriesID == target_series)[0]
    if len(series_idx) == 0:
        raise ValueError(f"Series {target_series} not found in model specification.")
    series_idx = series_idx[0]

    # 2. Extract Latent Data from Kalman Filter Results
    Res = res_object.result
    # C is the Lambda Matrix [N_series x N_factors]
    C = Res["C"]
    # F is the smoothed Latent Factor Matrix [T_time x N_factors]
    F = Res["Z"]  # DFM output aliases Latent Factors to 'Z'

    target_lambda = C[series_idx, :]

    # 3. Associate Factor Columns with Logical Blocks
    block_names = spec.BlockNames
    blocks_matrix = spec.Blocks  # [N_series x N_blocks]

    # Normally one factor per block unless explicitly customized
    # Factor assignment maps column-to-column conceptually
    block_contributions = {}

    if len(block_names) > target_lambda.shape[0]:
        raise ValueError(
            "Cannot decompose: Spec Config blocks mismatch factor resolution."
        )

    for i, block_name in enumerate(block_names):
        # Contribution over time = specific Lambda * specific Factor
        contribution = F[:, i] * target_lambda[i]
        block_contributions[block_name] = contribution

    df_contrib = pd.DataFrame(block_contributions)

    # AR(1) or idiosyncratic noise
    # Standard decomposition logic assigns unstructured variance to the residual
    df_contrib["Idiosyncratic Noise"] = (
        Res["X_sm"][:, series_idx] - df_contrib.sum(axis=1) - Res["Mx"][series_idx]
    )

    # Plotly Stacked Bar Chart
    fig = go.Figure()

    # Generate distinct colors for the blocks dynamically
    colors = px.colors.qualitative.Pastel

    for i, col in enumerate(df_contrib.columns):
        fig.add_trace(
            go.Bar(
                x=time_array,
                y=df_contrib[col],
                name=col,
                marker_color=colors[i % len(colors)],
            )
        )

    # Overlay the actual scaled prediction
    predicted_signal = (
        Res["X_sm"][:, series_idx] - Res["Mx"][series_idx]
    )  # Demeaned signal

    fig.add_trace(
        go.Scatter(
            x=time_array,
            y=predicted_signal,
            name="Total Demeaned Signal",
            mode="lines",
            line=dict(color="black", width=2),
        )
    )

    fig.update_xaxes(title_text="Date")
    fig.update_layout(
        barmode="relative",
        title=f"Structural Block Contributions Over Time: {target_series}",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
    )

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    return fig
