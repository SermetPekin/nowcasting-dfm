"""
Step 4 — Nowcast update (high-level API)
Shows how to quantify the impact of a new data release on the GDP nowcast.
Uses the sp_update_nowcast convenience wrapper — the easiest starting point.
See 04b_nowcast_update_core_api.py for the equivalent low-level approach.
"""

from dfm_sp import Options, sp_update_nowcast

series = "GDPC1"
period = "2016q4"

vintage_old = "2016-12-16"
vintage_new = "2016-12-23"

options_baseline = Options(
    vintage=vintage_old,
    max_iter=5000,
    spec_file_name="Spec_US_example.xls",
    threshold=1e-4,
    country="US",
    sample_start="2000-01-01",
    use_cache=True,
)

result_dict = sp_update_nowcast(options_baseline, vintage_new, series, period)

print(result_dict.keys())
# result_dict["fig"].show()  # interactive Plotly waterfall chart
