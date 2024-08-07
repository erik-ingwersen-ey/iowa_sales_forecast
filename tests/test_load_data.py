"""
Unit Tests for functions from `load_data` module.

Functions Tested
----------------
- `parse_combined_string(combined: str) -> dict`: Parses a combined string specifying the
  offset into its component parts (years, months, weeks, days).

- `create_date_offset_from_parts(years=0, months=0, weeks=0, days=0) -> pd.DateOffset`:
  Creates a `DateOffset` object from individual time components.

- `date_offset(n: int = None, freq: str = None, combined: str = None) -> pd.DateOffset`:
  Generates a `DateOffset` object based on either a combined string or separate `n` and
  `freq` parameters.

Test Cases
----------
- `test_parse_combined_string_valid()`: Tests valid combined strings to ensure they are
  parsed correctly.

- `test_parse_combined_string_invalid()`: Tests invalid combined strings to ensure they
  raise a `ValueError`.

- `test_create_date_offset_from_parts()`: Tests the creation of `DateOffset` objects from
  individual components.

- `test_date_offset_combined()`: Tests the `date_offset` function with valid combined
  strings.

- `test_date_offset_separate()`: Tests the `date_offset` function with separate `n` and
  `freq` parameters for each valid frequency.

- `test_date_offset_invalid_freq()`: Tests the `date_offset` function with an invalid
  frequency to ensure it raises a `ValueError`.

- `test_date_offset_missing_parameters()`: Tests the `date_offset` function with missing
  parameters to ensure it raises a `ValueError`.

How to Run
----------
Use the `pytest` framework to run these tests. Execute the following command in your terminal:

.. code-block:: console

    pytest <name_of_this_test_module>.py


These tests ensure that the `date_offset` function and its helpers work correctly and handle
various edge cases and invalid inputs gracefully.
"""
import pytest
import pandas as pd

from iowa_forecast.load_data import (
    parse_combined_string,
    create_date_offset_from_parts,
    date_offset,
)


def test_parse_combined_string_valid():
    assert parse_combined_string("2Y3M2W1D") == {
        "years": 2,
        "months": 3,
        "weeks": 2,
        "days": 1,
    }
    assert parse_combined_string("1Y") == {
        "years": 1,
        "months": 0,
        "weeks": 0,
        "days": 0,
    }
    assert parse_combined_string("4M2W") == {
        "years": 0,
        "months": 4,
        "weeks": 2,
        "days": 0,
    }


def test_parse_combined_string_invalid():
    with pytest.raises(ValueError):
        parse_combined_string("invalid")
    with pytest.raises(ValueError):
        parse_combined_string("5Z")


def test_create_date_offset_from_parts():
    assert create_date_offset_from_parts(
        years=1, months=2, weeks=3, days=4
    ) == pd.DateOffset(years=1, months=2, weeks=3, days=4)
    assert create_date_offset_from_parts() == pd.DateOffset(
        years=0, months=0, weeks=0, days=0
    )


def test_date_offset_combined():
    assert date_offset("2Y3M2W1D") == pd.DateOffset(years=2, months=3, weeks=2, days=1)
    assert date_offset("1Y") == pd.DateOffset(years=1, months=0, weeks=0, days=0)


def test_date_offset_separate():
    assert date_offset(5, "days") == pd.DateOffset(days=5, weeks=0, months=0, years=0)
    assert date_offset(2, "weeks") == pd.DateOffset(days=0, weeks=2, months=0, years=0)
    assert date_offset(3, "months") == pd.DateOffset(days=0, weeks=0, months=3, years=0)
    assert date_offset(1, "years") == pd.DateOffset(days=0, weeks=0, months=0, years=1)


def test_date_offset_invalid_freq():
    with pytest.raises(ValueError):
        date_offset(5, "invalid")


def test_date_offset_missing_parameters():
    with pytest.raises(ValueError):
        date_offset(5)
    with pytest.raises(ValueError):
        date_offset("days")
    with pytest.raises(ValueError):
        date_offset()


if __name__ == "__main__":
    pytest.main()
