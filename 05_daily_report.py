"""
Step 5 — Daily report
Runs the full DFM pipeline for the most recent available vintage and
writes an HTML report plus a multi-sheet Excel workbook to out/.
Intended for scheduled / recurring execution (e.g. cron, Task Scheduler).
"""

from pathlib import Path

from dfm_sp import Options, daily_report

out_folder = Path(".") / "out"
root = Path(".")

plot1_series = ("INDPRO", "HOUST", "PAYEMS", "CPIAUCSL", "UNRATE")
plot2_series = (("INDPRO", "HOUST"), ("PAYEMS", "CPIAUCSL"), ("UNRATE", "INDPRO"))

options = Options(
    root=root,
    max_iter=5000,
    threshold=1e-3,
    spec_file_name="Spec_US_example.xls",
    country="US",
    sample_start="2005-01-01",
    vintage="auto",
    plot1_series=plot1_series,
    plot2_series=plot2_series,
    out_folder=out_folder,
    use_cache=False,
)

print(options)
print("Running daily report...")
daily_report(options, write=True)
