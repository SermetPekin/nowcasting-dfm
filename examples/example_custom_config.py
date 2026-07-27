from pathlib import Path
import numpy as np

from dfm_sp import SpecConfig, LoadSpec
from dfm_sp import load_data
from dfm_sp import dfm
from dfm_sp.sp_utils import get_latest


def run_custom_spec_example():
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

    print("\n2. Loading Raw Data against Custom Spec...")
    # Locate the latest vintage data to test against
    data_folder = Path("data/US")
    latest_vintage = get_latest(data_folder)
    data_file = data_folder / f"{latest_vintage}.xls"

    # load_data automatically handles alignment, sorting, transformation, and ragged edges
    # based strictly on the parameters requested in the `spec` object!
    X, Time, Z = load_data(str(data_file), spec)

    print(f"Transformed Training Panel Shape: {X.shape}")
    print(f"Features mapped: {spec.SeriesID}")

    print("\n3. Running the compiled Numba JIT Expectation-Maximization Pipeline...")
    # Running for a short 50 iterations as an example
    Res = dfm(X, spec, threshold=1e-4, max_iter=50)

    print("\n✅ Custom Config DFM Run Complete!")
    print(f"Total Iterations Reached: {len(Res['loglik'])}")
    print(f"Final Log-Likelihood: {Res['loglik'][-1]:.2f}")


if __name__ == "__main__":
    run_custom_spec_example()
