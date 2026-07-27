from pathlib import Path
import numpy as np
import pandas as pd

from dfm_sp import SpecConfig, LoadSpec
from dfm_sp.core.load_data_pandas import load_data_pandas
from dfm_sp import dfm
from dfm_sp.sp_utils import get_latest


def run_pandas_spec_example():
    print("1. Defining SpecConfig entirely in Python...")
    # This demonstrates bypassing Excel entirely to declare your DFM blocks matrix,
    # series transformations, and variables locally.

    my_config = SpecConfig(
        series_id=["INDPRO", "PAYEMS", "CPIAUCSL", "UNRATE"],
        series_name=[
            "Industrial Production",
            "Payroll Employment",
            "Consumer Price Index",
            "Unemployment Rate",
        ],
        frequency=["m", "m", "m", "m"],
        units=["Index", "Thousands", "Index", "Percent"],
        transformation=[
            "pch",
            "chg",
            "pch",
            "chg",
        ],  # Different math required for each series
        category=["Real Activity", "Labor", "Prices", "Labor"],
        block_names=["Global", "RealActivity", "LaborMarket"],
        # Rows = Series, Columns = Blocks (Global must always be Col 0 = 1)
        blocks_matrix=np.array(
            [
                [1, 1, 0],  # INDPRO  -> Global & RealActivity
                [1, 0, 1],  # PAYEMS  -> Global & LaborMarket
                [1, 0, 0],  # CPI     -> Global Only
                [1, 0, 1],  # UNRATE  -> Global & LaborMarket
            ]
        ),
    )

    # Compile the config into the structured object the DFM algorithm expects
    spec = LoadSpec(my_config)

    print(
        "\n2. Connecting a raw Pandas DataFrame against Custom Spec (Bypassing File Loading)..."
    )
    # For demonstration, we simply load the csv/excel into pandas here, but in production,
    # this DataFrame could come from an API (e.g. FRED API), SQL Database, or pipeline stream.
    data_folder = Path("data/US")
    latest_vintage = get_latest(data_folder)
    data_file = data_folder / f"{latest_vintage}.xls"

    df = pd.read_excel(data_file)
    print(f"Loaded DataFrame with Shape: {df.shape}")

    # NEW: Push dataframe directly. Handles alignment, sorting, transformation, and ragged edges
    # based strictly on the parameters requested in the `spec` object!
    X, Time, Z = load_data_pandas(df, spec, date_col="Date")

    print(f"Transformed Training Panel Shape: {X.shape}")
    print(f"Features mapped: {spec.SeriesID}")

    print("\n3. Running the compiled Numba JIT Expectation-Maximization Pipeline...")
    # Running for a short 50 iterations as an example
    Res = dfm(X, spec, threshold=1e-4, max_iter=50)

    print("\n✅ DataFrame + Custom Config DFM Run Complete!")
    print(f"Total Iterations Reached: {len(Res['loglik'])}")
    print(f"Final Log-Likelihood: {Res['loglik'][-1]:.2f}")


if __name__ == "__main__":
    run_pandas_spec_example()
