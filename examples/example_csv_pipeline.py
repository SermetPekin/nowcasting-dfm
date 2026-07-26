from pathlib import Path
import pandas as pd
from dfm_sp import load_spec
from dfm_sp import load_data
from dfm_sp import dfm
from dfm_sp.sp_utils import get_latest


def main():
    # 1. Provide CSV Spec
    spec = load_spec("Spec_US_example.csv")

    # Generate a matching CSV for the data as well to test fully CSV pipeline
    data_folder = Path("data/US")
    latest_vintage = get_latest(data_folder)
    excel_data = data_folder / f"{latest_vintage}.xls"
    csv_data = data_folder / f"{latest_vintage}.csv"

    # Save the data to CSV
    pd.read_excel(excel_data).to_csv(csv_data, index=False)

    # 2. Provide CSV Data
    X, Time, Z = load_data(str(csv_data), spec)

    # 3. Predict Numba
    Res = dfm(X, spec, threshold=1e-4, max_iter=5)
    print("CSV->CSV Pipeline Success!")


if __name__ == "__main__":
    main()
