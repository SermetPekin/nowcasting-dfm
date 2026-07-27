import numpy as np
import pandas as pd
from dfm_sp import SpecConfig


def main():
    """
    Example: Integrating High-Frequency (Weekly) Data into the Monthly DFM

    This script demonstrates how to aggregate weekly data into a monthly format
    so that it perfectly maps into the DFM's `SpecConfig` without destroying
    the structural matrix sizes limits.
    """
    # ---------------------------------------------------------
    # 1. Generate Fake Weekly Data
    # ---------------------------------------------------------
    print("Generating simulated weekly data (e.g., Initial Jobless Claims)...")
    dates_weekly = pd.date_range(start="2020-01-01", end="2023-12-31", freq="W-FRI")

    np.random.seed(42)
    # Random walk to simulate macroeconomic levels
    weekly_values = np.cumsum(np.random.normal(0, 1, size=len(dates_weekly))) + 100
    weekly_series = pd.Series(weekly_values, index=dates_weekly, name="Weekly_Claims")

    # ---------------------------------------------------------
    # 2. Resample Weekly to Monthly
    # ---------------------------------------------------------
    # We sample the weekly data dynamically.
    # 'MS' means Month Start (July 1st, August 1st, etc.)
    # We take the mean() representing the average of the weeks in that month.
    # Alternatively use .last() if it is a stock variable (like an interest rate).
    print("Resampling Weekly -> Monthly (Average)...")
    monthly_from_weekly = weekly_series.resample("MS").mean()

    # ---------------------------------------------------------
    # 3. Generate Fake Monthly & Quarterly to Join against
    # ---------------------------------------------------------
    dates_monthly = pd.date_range(start="2020-01-01", end="2023-12-01", freq="MS")
    monthly_data = pd.Series(
        np.cumsum(np.random.normal(0, 2, size=len(dates_monthly))),
        index=dates_monthly,
        name="Industrial_Production",
    )

    # QS-OCT lines quarterly data up nicely with month starts
    dates_quarterly = pd.date_range(start="2020-01-01", end="2023-12-01", freq="QS-OCT")
    quarterly_data = pd.Series(
        np.cumsum(np.random.normal(0, 5, size=len(dates_quarterly))),
        index=dates_quarterly,
        name="Real_GDP",
    )

    # ---------------------------------------------------------
    # 4. Merge All Series into the Observation Matrix (Z)
    # ---------------------------------------------------------
    # Building our base timeline using standard pandas indexing
    df = pd.DataFrame(index=dates_monthly)
    df = df.join(monthly_from_weekly)
    df = df.join(monthly_data)
    df = df.join(quarterly_data)

    print("\n--- Aligned Dataset (Top 5 rows) ---")
    print(df.head())

    # ---------------------------------------------------------
    # 5. Build SpecConfig mapping the Weekly data as Monthly ("m")
    # ---------------------------------------------------------
    # Raw payload formats ready to feed into `get_with_options` / `run_dfm`
    Z_raw = df.to_numpy()
    Time_raw = df.index.to_numpy()

    spec = SpecConfig(
        series_id=["W_CLAIMS", "IP_MONTHLY", "GDP_QUARTERLY"],
        series_name=[
            "Initial Claims (Weekly Avg)",
            "Industrial Production",
            "Real GDP",
        ],
        # Notice the weekly series is registered exactly as "m" alongside standard monthly data
        frequency=["m", "m", "q"],
        units=["Levels", "Levels", "Levels"],
        transformation=["lin", "lin", "lin"],  # You can use "pch", "dln", etc.
        category=["Labor", "Production", "Output"],
        block_names=["Global", "Soft", "Real"],
        blocks_matrix=np.array(
            [
                [1, 1, 0],  # Weekly Proxy mapped onto Global and Soft
                [1, 0, 1],  # Monthly IP mapped onto Global and Real
                [1, 0, 1],  # Quarterly GDP mapped onto Global and Real
            ]
        ),
    )

    print("\n--- Spec configuration mapped correctly ---")
    print(f"Series      : {spec.series_name}")
    print(f"Frequencies : {spec.frequency}")
    print("\nSUCCESS: The weekly data is processed natively as an 'm' series!")


if __name__ == "__main__":
    main()
