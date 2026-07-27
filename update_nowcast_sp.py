from dfm_sp import sp_update_nowcast, Options

series = "GDPC1"
period = "2016q4"
#
vintage_old = "2016-12-16"
vintage_new = "2016-12-23"
sample_start = "2000-01-01"
country = "US"
spec_file_name = "Spec_US_example.xls"
max_iter = 5000

options_baseline = Options(
    vintage=vintage_old,
    max_iter=max_iter,
    spec_file_name=spec_file_name,
    threshold=1e-4,
    country=country,
    sample_start=sample_start,
    use_cache=True,
)

result_dict = sp_update_nowcast(options_baseline, vintage_new, series, period)

print(result_dict.keys())
