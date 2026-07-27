"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from dfm_sp.sp_plots import (
    plot_loglik,
    plot_common,
    plot_projection_x_over_y,
    plot_transformed_data,
)
from dfm_sp.sp_plots_blocks import plot_block_contributions
from dfm_sp.sp_news import plot_news_waterfall

from dfm_sp.sp_plots import (
    plot_factor_contribution,
    plot_factors_with_series,
    plot_prediction_intervals,
)
from dfm_sp.sp_heatmap import plot_factor_loadings
from io import BytesIO
import base64


def generate_html_report(ResObject, Time, X, Z, options, output_path=None):
    Spec = ResObject.spec
    OUT_FOLDER = options.out_folder
    OUT_FOLDER.mkdir(exist_ok=True)
    if output_path is None:
        output_path = OUT_FOLDER / f"report-{options.name_format()}.html"
    SHOW = False
    html_content = [
        f"<h1>DFM Analysis Report </h1> <h3> Vintage: {options.vintage_date} - max_iter : {options.max_iter } <h3>"
    ]
    # Single plots
    plots = [
        ("Common Components", plot_common(ResObject, Time, show=SHOW)),
        ("Log-Likelihood", plot_loglik(ResObject, Time, show=SHOW)),
        ("Prediction Intervals", plot_prediction_intervals(ResObject, Time, show=SHOW)),
        ("Factor Contribution", plot_factor_contribution(ResObject, show=SHOW)),
        ("Factor Loadings", plot_factor_loadings(ResObject, show=SHOW)),
        # Plot Block Contributions specifically passing the Spec object and a default target (first target series if any)
        (
            "Block Contributions",
            (
                plot_block_contributions(
                    ResObject,
                    Spec,
                    Time,
                    (
                        options.target_series
                        if getattr(options, "target_series", None)
                        else Spec.SeriesID[0]
                    ),
                )
                if hasattr(plot_block_contributions, "__call__")
                else None
            ),
        ),
        (
            "Factors with Series",
            plot_factors_with_series(ResObject, Time, options.plot1_series, show=SHOW),
        ),
    ]
    for title, fig in plots:
        if fig:
            if hasattr(fig, "to_html"):  # Plotly figure
                html_content.append(f"<h2>{title}</h2>")
                html_content.append(fig.to_html(full_html=False))
            else:  # Matplotlib figure
                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode("utf-8")
                html_content.append(
                    f"<h2>{title}</h2><img src='data:image/png;base64,{img_str}'>"
                )
    for item in options.plot1_series:
        figs = plot_transformed_data(Spec, X, Z, Time, [item], show=SHOW)
        if isinstance(figs, list) and len(figs) > 0:
            fig = figs[0]
            if fig:
                if hasattr(fig, "to_html"):
                    html_content.append(
                        f'<div class="plot"><h2>Raw vs Transformed: {item}</h2>'
                    )
                    html_content.append(
                        fig.to_html(full_html=False, include_plotlyjs="cdn")
                    )
                    html_content.append("</div>")
                else:
                    buf = BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode("utf-8")
                    html_content.append(
                        f'<div class="plot"><h2>Raw vs Transformed: {item}</h2>'
                        f'<img src="data:image/png;base64,{img_str}"></div>'
                    )
    for items in options.plot2_series:
        fig = plot_projection_x_over_y(ResObject, X, Z, Time, items, show=SHOW)
        if fig:
            if hasattr(fig, "to_html"):  # Plotly figure
                html_content.append(f"<h2>Projection: {items}</h2>")
                html_content.append(fig.to_html(full_html=False))
            else:  # Matplotlib figure
                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode("utf-8")
                html_content.append(
                    f"<h2>Projection: {items}</h2><img src='data:image/png;base64,{img_str}'>"
                )
    with open(output_path, "w", encoding="utf-8", errors="replace") as f:
        f.write("\n".join(html_content))
    print(f"HTML report saved to {output_path}")
