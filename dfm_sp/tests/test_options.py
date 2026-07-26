import pytest
from pathlib import Path
from dfm_sp.sp_classes import Options


def test_options_copy_method():
    """Test that Options.copy() creates a deep enough copy and handles vintage assignment correctly."""
    opt_base = Options(
        vintage="2016-12-16",
        country="US",
        spec_file_name="Spec_US_example.xls",
        max_iter=5000,
    )

    assert opt_base.vintage == "2016-12-16"
    assert opt_base.vintage_date == "2016-12-16"

    # Create copy using the new method
    opt_new = opt_base.copy()

    # Modify the vintage property in the copy
    opt_new.vintage = "2016-12-23"

    # Test independence of properties between base and new instance
    assert opt_base.vintage == "2016-12-16", "Base vintage should not change"
    assert opt_new.vintage == "2016-12-23", "Copied vintage should update"

    # Test that @property vintage_date updates dynamically per instance
    assert (
        opt_base.vintage_date == "2016-12-16"
    ), "Base vintage_date should remain original"
    assert (
        opt_new.vintage_date == "2016-12-23"
    ), "Copied vintage_date should evaluate to new vintage"

    # Test that base properties transferred safely
    assert opt_new.country == "US"
    assert opt_new.max_iter == 5000

    # Test hash keys are different since vintage parameter drives identical parameter states
    # Note: frozen hash state generation runs on __post_init__ so we test properties directly
