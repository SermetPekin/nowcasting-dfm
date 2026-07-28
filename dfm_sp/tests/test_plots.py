"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

Integration tests for all plotting functions in sp_plots.py.
Uses the session-scoped pipeline_result fixture from conftest.py so
the EM run is not repeated.
All plots are generated with show=False to avoid opening a browser.
"""

import pytest
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SPEC_FILE = PROJECT_ROOT / "Spec_US_example.xls"
DATA_FILE = PROJECT_ROOT / "data" / "US" / "2016-06-29.xls"

pytestmark = pytest.mark.skipif(
    not SPEC_FILE.exists() or not DATA_FILE.exists(),
    reason="Sample data not present — run download_sample_data() first.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_figure(fig, min_traces: int = 1):
    """Assert obj is a non-empty Plotly Figure."""
    assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"
    assert (
        len(fig.data) >= min_traces
    ), f"Expected at least {min_traces} trace(s), got {len(fig.data)}"


# ---------------------------------------------------------------------------
# Data transformation plots
# ---------------------------------------------------------------------------


def test_plot_transformed_data_one(pipeline_result):
    from dfm_sp.sp_plots import plot_transformed_data_one

    _, Spec, X, Time, Z, _ = pipeline_result
    series = Spec.SeriesID[0]
    fig = plot_transformed_data_one(Spec, X, Z, Time, series_id=series, show=False)
    _assert_figure(fig, min_traces=2)  # raw + transformed subplots


def test_plot_transformed_data(pipeline_result):
    from dfm_sp.sp_plots import plot_transformed_data

    _, Spec, X, Time, Z, _ = pipeline_result
    series = list(Spec.SeriesID[:2])
    figs = plot_transformed_data(Spec, X, Z, Time, series_wanted=series, show=False)
    assert isinstance(figs, list)
    assert len(figs) == 2
    for fig in figs:
        _assert_figure(fig, min_traces=2)


# ---------------------------------------------------------------------------
# Model / convergence plots
# ---------------------------------------------------------------------------


def test_plot_loglik(pipeline_result):
    from dfm_sp.sp_plots import plot_loglik

    _, _, _, Time, _, res_object = pipeline_result
    fig = plot_loglik(res_object, Time, show=False)
    _assert_figure(fig)


def test_plot_common(pipeline_result):
    from dfm_sp.sp_plots import plot_common

    _, Spec, _, Time, _, res_object = pipeline_result
    series = Spec.SeriesID[0]
    fig = plot_common(res_object, Time, series_id=series, show=False)
    _assert_figure(fig)


def test_plot_common_invalid_series(pipeline_result):
    from dfm_sp.sp_plots import plot_common

    _, _, _, Time, _, res_object = pipeline_result
    with pytest.raises(ValueError, match="not found"):
        plot_common(res_object, Time, series_id="__NONEXISTENT__", show=False)


def test_plot_loglik_together(pipeline_result):
    from dfm_sp.sp_plots import plot_loglik_together

    _, Spec, _, Time, _, res_object = pipeline_result
    if "INDPRO" not in Spec.SeriesID:
        pytest.skip("INDPRO not in spec — required by plot_loglik_together")
    fig_ll, fig_common = plot_loglik_together(res_object, Time, show=False)
    _assert_figure(fig_ll)
    _assert_figure(fig_common)


def test_plot_projection_x_over_y(pipeline_result):
    from dfm_sp.sp_plots import plot_projection_x_over_y

    _, Spec, X, Time, Z, res_object = pipeline_result
    series = list(Spec.SeriesID[:2])
    fig = plot_projection_x_over_y(res_object, X, Z, Time, series=series, show=False)
    _assert_figure(fig, min_traces=2)


def test_plot_projection_invalid_series(pipeline_result):
    from dfm_sp.sp_plots import plot_projection_x_over_y

    _, _, X, Time, Z, res_object = pipeline_result
    with pytest.raises(ValueError, match="not found"):
        plot_projection_x_over_y(
            res_object, X, Z, Time, series=["__NONEXISTENT__"], show=False
        )


# ---------------------------------------------------------------------------
# Factor analysis plots
# ---------------------------------------------------------------------------


def test_plot_factors_with_series(pipeline_result):
    from dfm_sp.sp_plots import plot_factors_with_series

    _, Spec, _, Time, _, res_object = pipeline_result
    series_list = list(Spec.SeriesID[:2])
    fig = plot_factors_with_series(
        res_object, Time, series_list=series_list, show=False
    )
    _assert_figure(fig)
    n_factors = res_object.result["Z"].shape[1]
    assert len(fig.data) == n_factors + len(series_list)


def test_plot_prediction_intervals(pipeline_result):
    from dfm_sp.sp_plots import plot_prediction_intervals

    _, Spec, _, Time, _, res_object = pipeline_result
    series = Spec.SeriesID[0]
    fig = plot_prediction_intervals(res_object, Time, target_series=series, show=False)
    _assert_figure(fig, min_traces=2)  # nowcast + interval band


def test_plot_prediction_intervals_invalid_series(pipeline_result):
    from dfm_sp.sp_plots import plot_prediction_intervals

    _, _, _, Time, _, res_object = pipeline_result
    with pytest.raises(ValueError, match="not found"):
        plot_prediction_intervals(
            res_object, Time, target_series="__NONEXISTENT__", show=False
        )


def test_plot_factor_contribution(pipeline_result):
    from dfm_sp.sp_plots import plot_factor_contribution

    _, Spec, _, _, _, res_object = pipeline_result
    fig = plot_factor_contribution(res_object, show=False)
    _assert_figure(fig)
    n_factors = res_object.result["C"].shape[1]
    assert len(fig.data) == n_factors  # one Bar trace per factor


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------


def test_plot_covariance_network(pipeline_result):
    from dfm_sp.sp_plots import plot_covariance_network

    _, Spec, _, _, _, res_object = pipeline_result
    fig = plot_covariance_network(res_object, show=False)
    _assert_figure(fig)
    heatmap = fig.data[0]
    assert heatmap.type == "heatmap"
    n = len(Spec.SeriesID)
    assert np.array(heatmap.z).shape == (n, n)


def test_plot_em_step_deltas(pipeline_result):
    from dfm_sp.sp_plots import plot_em_step_deltas

    _, _, _, _, _, res_object = pipeline_result
    fig = plot_em_step_deltas(res_object, show=False)
    _assert_figure(fig)
    ll = np.array(res_object.result["loglik"])
    assert len(fig.data[0].x) == len(ll) - 1  # one delta per adjacent pair
