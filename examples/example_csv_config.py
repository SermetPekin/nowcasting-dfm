from dfm_sp import LoadSpec


def test_csv_load():
    csv_file = "Spec_US_example.csv"
    spec = LoadSpec(csv_file)
    print(f"Successfully loaded CSV spec with {len(spec.SeriesID)} series.")
    print(f"Categories present: {list(set(spec.Category))}")


if __name__ == "__main__":
    test_csv_load()
