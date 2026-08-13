"""
Step 4b — Nowcast update (core API)
Same goal as 04_nowcast_update.py but uses the lower-level core functions
directly — useful if you need fine-grained control over the update step.
"""

import os

from dfm_sp import Options, get_with_options, run
from dfm_sp import load_data
from dfm_sp import update_nowcast

series = "GDPC1"  # Nowcasting real GDP
period = "2016q4"

vintage_old = "2016-12-16"
vintage_new = "2016-12-23"

options_baseline = Options(
    vintage=vintage_old, max_iter=5000, threshold=1e-4, use_cache=True
)

print(f"Loading Base Line Model for vintage: {vintage_old}")
Spec, X_old, _, Z_old = get_with_options(options_baseline)

ResObject = run(X_old, Spec, options_baseline)
ResObject.write()
Res = ResObject.result

print(f"Loading Updated Data for vintage: {vintage_new}")
datafile_new = os.path.join("data", options_baseline.country, vintage_new + ".xls")

X_new, Time, _ = load_data(datafile_new, Spec, options_baseline.sample_start)

update_nowcast(X_old, X_new, Time, Spec, Res, series, period, vintage_old, vintage_new)
