"""
General utility functions.
"""
import pandas as pd
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


def create_bigquery_table_from_pandas(client: bigquery.Client, dataframe: pd.DataFrame, table_id: str, dataset_id="bqmlforecast"):
    table_ref = client.dataset(dataset_id).table(table_id)
    load_job = client.load_table_from_dataframe(dataframe, table_ref)
    load_job.result()
