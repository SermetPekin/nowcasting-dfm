"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""


import pandas as pd
import plotly.graph_objects as go


def plot_news_waterfall(
    news_table: pd.DataFrame,
    y_old: float,
    y_new: float,
    vintage_old: str,
    vintage_new: str,
    target_series: str,
) -> go.Figure:
    """
    Generates a Plotly Waterfall chart showcasing the news attribution from the Kalman Filter update.

    Args:
        news_table: DataFrame containing 'Impact' of each released series. Must contain 'Impact' column and index/SeriesName.
        y_old: The previous Nowcast prediction.
        y_new: The updated Nowcast prediction.
        vintage_old: Date string of the old vintage.
        vintage_new: Date string of the new vintage.
        target_series: The name of the series being plotted (e.g. GDP).
    """
    # Filter only series that had an actual non-zero impact.
    # A tiny threshold is used to prevent graphing noise.
    impacts = news_table[abs(news_table["Impact"]) > 1e-5]["Impact"]

    # We also might have 'Revisions' if past data changed, but typically we focus on 'Impact'
    # For a perfect waterfall math: Old + Sum(Impacts) + (any revisions) = New

    # Calculate the un-attributed residual (usually historical revisions or nonlinearities)
    sum_impacts = impacts.sum()
    residual = (y_new - y_old) - sum_impacts

    # Build Waterfall Arrays
    measure = ["absolute"]
    x_labels = [f"Old Nowcast<br>({vintage_old})"]
    y_values = [y_old]

    for series_name, impact in impacts.items():
        measure.append("relative")
        # Truncate long names for rendering
        label = (
            str(series_name)[:20] + "..."
            if len(str(series_name)) > 20
            else str(series_name)
        )
        x_labels.append(label)
        y_values.append(impact)

    if abs(residual) > 1e-4:
        measure.append("relative")
        x_labels.append("Other Revisions")
        y_values.append(residual)

    measure.append("total")
    x_labels.append(f"New Nowcast<br>({vintage_new})")
    y_values.append(
        y_new
    )  # Waterfall "total" automatically sums up, but we assign y_new for clarity

    fig = go.Figure(
        go.Waterfall(
            name="News",
            orientation="v",
            measure=measure,
            x=x_labels,
            textposition="outside",
            text=[
                f"{val:+.3f}" if m == "relative" else f"{val:.3f}"
                for m, val in zip(measure, y_values)
            ],
            y=y_values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#EF553B"}},
            increasing={"marker": {"color": "#00CC96"}},
            totals={"marker": {"color": "#636EFA"}},
        )
    )

    fig.update_layout(
        title=f"Nowcast News Attribution: {target_series}",
        showlegend=False,
        waterfallgap=0.3,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Add grid lines
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    return fig
