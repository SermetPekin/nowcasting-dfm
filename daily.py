"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""

from dfm_sp import Path, Options, daily_report

out_folder = Path(".") / "out "
root = Path(".")

country = "US"
max_iter = 2
threshold = 1e-3
spec_file_name = "Spec_US_example.xls"
sample_start = "2005-01-01"

vintage = "auto"

plot1_series = ("INDPRO", "HOUST", "PAYEMS", "CPIAUCSL", "UNRATE")
plot2_series = (("INDPRO", "HOUST"), ("PAYEMS", "CPIAUCSL"), ("UNRATE", "INDPRO"))

options = Options(
    root=root,
    max_iter=max_iter,
    threshold=threshold,
    spec_file_name=spec_file_name,
    country=country,
    sample_start=sample_start,
    vintage=vintage,
    plot1_series=plot1_series,
    plot2_series=plot2_series,
    out_folder=out_folder,
)

print(options)

print("Running daily report...")
daily_report(options)
