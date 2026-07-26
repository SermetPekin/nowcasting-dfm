from pathlib import Path
from dfm_sp import load_spec


def test_csv_load():
    csv_file = "Spec_US_example.csv"
    spec = load_spec(csv_file)
    print(f"Successfully loaded CSV spec with {len(spec.SeriesID)} series.")
    print(f"Categories present: {list(set(spec.Category))}")


if __name__ == "__main__":
    test_csv_load()
