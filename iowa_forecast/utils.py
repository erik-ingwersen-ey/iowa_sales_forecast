"""
General utility functions.
"""
from __future__ import annotations

from typing import List, Tuple
import re

import pandas as pd
import fnmatch

import google
from google.cloud import bigquery


def normalize_item_name(item_name: str) -> str:
    """
    Convert 'item_name' values to lower case and replace spaces with underscores.

    Parameters
    ----------
    item_name : str
        Item names to normalize.

    Returns
    -------
    str
        Normalized item names.

    Examples
    --------
    >>> normalize_item_name("TITOS HANDMADE VODKA")
    'titos_handmade_vodka'

    Notes
    -----
    Used to generate names for the different ARIMA models that are created for each
    unique item name.
    """
    return item_name.lower().replace(' ', '_')


def split_table_name_info(table_name: str) -> Tuple[str | None, str | None, str]:
    """
    Extract components from a table name.

    Parameters
    ----------
    table_name : str
        Table name to extract components from.

    Returns
    -------
    Tuple[str | None, str | None, str]
        A tuple containing the project ID, dataset ID and table name if any of
        these components are in the table name. If one of the components is not
        contained inside `table_name`, then they are returned as None.

    Examples
    --------
    >>> split_table_name_info('my_project.my_dataset.my_table')
    ('my_project', 'my_dataset', 'my_table')
    >>> split_table_name_info('my_dataset.my_table')
    (None, 'my_dataset', 'my_table')
    >>> split_table_name_info('my_table')
    (None, None, 'my_table')
    """
    table_components = table_name.split(".")
    if len(table_components) == 1:
        return None, None, table_components[0]
    if len(table_components) == 2:
        return None, table_components[0], table_components[1]
    if len(table_components) == 3:
        return table_components[0], table_components[1], table_components[2]
    raise ValueError(
        f"Table name contains more than three components: {table_name}"
    )


def create_bigquery_table_from_pandas(
    client: bigquery.Client,
    dataframe: pd.DataFrame,
    table_id: str,
    dataset_id="bqmlforecast",
    if_exists: str = "replace",
):
    """Create a BigQuery table from a pandas DataFrame.

    Parameters
    ----------
    client : bigquery.Client
        BigQuery client used to connect to the service.
    dataframe : pd.DataFrame
        A `pandas.DataFrame` to load into the BigQuery table.
    table_id : str
        ID of the table to create in BigQuery.
    dataset_id : str, default="bqmlforecast"
        ID of the dataset where the table will be created.
    if_exists : {"fail", "replace", "append"}, default="replace"
        Behavior when the table already exists.

    Examples
    --------
    >>> client = bigquery.Client()
    >>> dataframe = pd.DataFrame({'column1': [1, 2], 'column2': ['a', 'b']})
    >>> create_bigquery_table_from_pandas(client, dataframe, 'my_table')
    """
    project_id = client.project
    if dataset_id in table_id:
        table_id = table_id.replace(f"{dataset_id}.", "")
    if project_id in table_id:
        table_id = table_id.replace(f"{project_id}.", "")

    _project_id, _dataset_id, table_id = split_table_name_info(table_id)
    dataset_id = _dataset_id if _dataset_id is not None else dataset_id
    project_id = _project_id if _project_id is not None else project_id

    table_ref = client.dataset(dataset_id).table(table_id)

    if if_exists == "replace":
        client.delete_table(table_ref, not_found_ok=True)
        load_job = client.load_table_from_dataframe(dataframe, table_ref)
    elif if_exists == "append":
        load_job = client.load_table_from_dataframe(dataframe, table_ref)
    else:
        if client.get_table(table_ref):
            raise ValueError(f"Table {table_id} already exists in dataset {dataset_id}")
        load_job = client.load_table_from_dataframe(dataframe, table_ref)

    load_job.result()


def create_dataset_if_not_found(
    client: bigquery.Client,
    project_id: str | None = None,
    dataset_name: str = "bqmlforecast",
    location: str = "us",
):
    """
    Create a BigQuery dataset if it does not exist.

    Parameters
    ----------
    client : bigquery.Client
        BigQuery client used to connect to the service.
    project_id : str, optional
        ID of the project where the dataset will be created.
        If no value is provided, the Project ID gets inferred from
        the `project` attibute from `client`.
    dataset_name : str, default="bqmlforecast"
        Name of the dataset to create.
    location : str, default="us"
        Location of the dataset.

    Raises
    ------
    Exception
        If any exception other than the error informing
        the dataset already exists.

    Examples
    --------
    >>> client = bigquery.Client()
    >>> create_dataset_if_not_found(client, dataset_name='new_dataset')
    Dataset 'new_dataset' already exists.

    Notes
    -----
    This function checks if the specified dataset exists in the given project. If it does
    not exist, the function creates the dataset.
    """
    if project_id is None:
        project_id = client.project

    # Construct a full Dataset object to send to the API
    dataset_id = f"{project_id}.{dataset_name}"
    dataset = bigquery.Dataset(dataset_id)  # noqa

    # Set the location
    dataset.location = location

    # Check if the dataset exists
    try:
        client.get_dataset(dataset_id)  # Make an API request.
        print(f"Dataset '{dataset_name}' already exists.")
    except Exception as exc:
        if isinstance(exc, google.api_core.exceptions.NotFound):  # noqa
            # Dataset does not exist, create it
            dataset = client.create_dataset(dataset)
            print(f"Created dataset '{dataset_name}'.")
        else:
            raise exc


def list_tables_with_pattern(
    client: bigquery.Client,
    dataset_id: str,
    table_pattern: str,
    project_id: str | None = None,
) -> List[str]:
    """
    List BigQuery tables matching a specific pattern.

    Constructs a fully qualified dataset ID, retrieves the dataset,
    lists all tables, and filters them based on the provided pattern.

    Parameters
    ----------
    client : bigquery.Client
        The BigQuery client used to interact with the service.
    dataset_id : str
        The ID of the dataset containing the tables to list.
    table_pattern : str
        The pattern to match against the table IDs.
    project_id : str, optional
        The ID of the project containing the dataset. If None,
        the client's project is used.

    Returns
    -------
    List[str]
        A list of table IDs that match the specified pattern.

    Notes
    -----
    The `fnmatch` module is used to filter tables based on the pattern.
    Ensure that the pattern provided is compatible with `fnmatch`.

    Examples
    --------
    List all tables in a dataset that match the pattern 'sales_*':

    >>> client = bigquery.Client()
    >>> tables = list_tables_with_pattern(client, 'my_dataset', 'sales_*')
    >>> print(tables)
    ['sales_2021', 'sales_2022']
    """

    # Construct the fully qualified dataset ID
    project = client.project if project_id is None else project_id
    dataset_ref = f"{project}.{dataset_id}"

    # Get the dataset
    dataset = client.get_dataset(dataset_ref)

    # List all tables in the dataset
    tables = client.list_tables(dataset)

    # Filter tables based on the pattern using fnmatch
    matching_tables = [
        table.table_id for table in tables if fnmatch.fnmatch(table.table_id, table_pattern)
    ]

    return matching_tables


def parse_combined_string(combined: str) -> dict:
    """Parse a combined offset string into its components.

    Parameters
    ----------
    combined : str
        A combined string specifying the offset, e.g., `'2Y3M2W1D'`.

    Returns
    -------
    dict
        A dictionary with keys `'years'`, `'months'`, `'weeks'`, `'days'`
        and their corresponding values.

    Raises
    ------
    ValueError
        If the combined string is invalid.
    """
    pattern = re.compile(
        r'(?P<years>\d+Y)?(?P<months>\d+M)?(?P<weeks>\d+W)?(?P<days>\d+D)?',
        re.IGNORECASE
    )
    match = pattern.fullmatch(combined)
    if not match:
        raise ValueError(f"The specified `combined` string {combined} is not valid.")

    return {k: int(v[:-1]) if v else 0 for k, v in match.groupdict().items()}


def create_date_offset_from_parts(years=0, months=0, weeks=0, days=0) -> pd.DateOffset:
    """Create a `pandas.DateOffset` object from individual time components.

    Parameters
    ----------
    years : int, default=0
        Number of years for the offset.
    months : int, default=0
        Number of months for the offset.
    weeks : int, default=0
        Number of weeks for the offset.
    days : int, default=0
        Number of days for the offset.

    Returns
    -------
    pd.DateOffset
        A `pandas.DateOffset` object for the specified time components.
    """
    return pd.DateOffset(years=years, months=months, weeks=weeks, days=days)


def date_offset(*args: Union[int, str], freq: str = None) -> pd.DateOffset:
    """
    Generate a `pandas.DateOffset` based on the given frequency and value or a combined string.

    Parameters
    ----------
    args : int or str
        * If one argument is provided, it should be a combined string specifying
          the offset, e.g., `'2Y3M2W1D'`.
        * If two arguments are provided, they should be `n` (int) and `freq` (str).
    freq : str {'days', 'weeks', 'months', 'years'}, optional
        The frequency type. Valid options are `'days'`, `'weeks'`, `'months'`, `'years'`.
        Ignored if `combined` is provided.

    Returns
    -------
    pd.DateOffset
        A `pandas.DateOffset` object for the specified frequency and value.

    Raises
    ------
    ValueError
        If `freq` is not one of the valid options or if the combined string is invalid.
    """
    if len(args) == 1 and isinstance(args[0], str):
        combined = args[0]
        offset_parts = parse_combined_string(combined)
        return create_date_offset_from_parts(**offset_parts)

    if len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], str):
        n, freq = args
        freq = freq.lower()
        valid_freqs = {"d": "days", "day": "days", "days": "days",
                       "w": "weeks", "week": "weeks", "weeks": "weeks",
                       "m": "months", "month": "months", "months": "months",
                       "y": "years", "year": "years", "years": "years"}

        if freq not in valid_freqs:
            raise ValueError(f"The specified `freq` {freq} is not a valid frequency. "
                             "Valid frequencies are: 'days', 'weeks', 'months', 'years'.")

        return create_date_offset_from_parts(**{valid_freqs[freq]: n})

    raise ValueError(
        "Either provide a single combined string or both `n` and `freq` as arguments.")
