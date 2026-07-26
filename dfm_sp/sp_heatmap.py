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
