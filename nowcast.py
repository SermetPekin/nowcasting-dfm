
import os
import pickle
from dfm_sp.core.load_data import load_data
from dfm_sp.core.load_spec import LoadSpec
from dfm_sp.core.update_Nowcast import update_nowcast
from dfm_sp import Options
from dfm_sp import get_with_options, run

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
Res = ResObject.result

print(f"Loading Updated Data for vintage: {vintage_new}")
datafile_new = os.path.join("data", options_baseline.country, vintage_new + ".xls")

X_new, Time, _ = load_data(datafile_new, Spec, options_baseline.sample_start)

update_nowcast(X_old, X_new, Time, Spec, Res, series, period, vintage_old, vintage_new)
