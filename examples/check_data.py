import numpy as np
from dfm_sp.sp_classes import Options
from dfm_sp.sp_run import get_with_options

options_old = Options(
    vintage="2016-12-16", country="US", spec_file_name="Spec_US_example.xls"
)
options_new = Options(
    vintage="2016-12-23", country="US", spec_file_name="Spec_US_example.xls"
)

Spec_old, X_old, _, _ = get_with_options(options_old)
Spec_new, X_new, _, _ = get_with_options(options_new)

print("X_old shape:", X_old.shape)
print("X_new shape:", X_new.shape)
print(
    "Are they exactly exactly the same? ", np.array_equal(X_old, X_new, equal_nan=True)
)
diff_count = np.sum(np.isnan(X_old) != np.isnan(X_new))
print(f"Number of cells where NaN status differs: {diff_count}")
