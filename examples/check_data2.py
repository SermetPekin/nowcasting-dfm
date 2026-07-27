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

miss_old = np.isnan(X_old)
miss_new = np.isnan(X_new)
i_miss = miss_old.astype(int) - miss_new.astype(int)
t_miss, v_miss = np.where(i_miss == 1)

print("v_miss dimensions:", v_miss.shape)
print("t_miss dimensions:", t_miss.shape)
print("i_miss counts:", np.unique(i_miss, return_counts=True))
