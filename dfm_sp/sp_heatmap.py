"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""

import plotly.graph_objects as go

# -------------------------------------------------Libraries
# Libs
from dfm_sp.sp_classes import ResultObject


def plot_factor_loadings(res_obj: ResultObject, show: bool = True) -> go.Figure:
    """Heatmap of factor loadings (C matrix)."""
    Res = res_obj.result
    Spec = res_obj.spec
    C = Res["C"]  # (n_series × n_factors)
    fig = go.Figure(
        go.Heatmap(
            z=C.T.copy(),  # Transpose to show factors as rows
            x=Spec.SeriesID,
            y=[f"Factor {i+1}" for i in range(C.shape[1])],
            colorscale="RdBu",
            zmid=0,
        )
    )
    fig.update_layout(
        title="Factor Loadings Heatmap",
        xaxis_title="Series",
        yaxis_title="Factor",
    )
    if show:
        fig.show()
    return fig
