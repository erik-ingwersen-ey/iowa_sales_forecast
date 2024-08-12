"""
BigQuery Model Training and Execution Module.

This module provides functionality for creating, executing, and managing
'ARIMA_PLUS_XREG' models using Google BigQuery. The module includes
functions to generate SQL queries for creating models, executing these
queries with retries, and evaluating model performance.

Functions
---------
* `create_model_query`: Generate an SQL query to create an ARIMA_PLUS_XREG model
  for a specified item and its associated data.

* `execute_query_with_retries`: Execute a given SQL query with retry logic in case of failure.

* `create_models_for_items`: Create 'ARIMA_PLUS_XREG' models for a list of items
  by executing generated SQL queries.

* `train_arima_models`: Train ARIMA models for specified columns,
  executes the corresponding SQL queries, and stores the model metrics in BigQuery tables.

Notes
-----
This module is designed to work with Google BigQuery and requires a valid
BigQuery client instance. The models generated by this module are intended
for forecasting time series data, with options to handle holiday effects,
step changes, and data cleaning.

See Also
--------
Google BigQuery: https://cloud.google.com/bigquery
BigQuery ML: https://cloud.google.com/bigquery-ml
"""
from __future__ import annotations

import time
from typing import List

from google.cloud import bigquery  # pylint: disable=no-name-in-module
from rich.progress import track

from iowa_forecast.models_configs import ARIMA_PLUS_XREG_Config, ARIMAConfig
from iowa_forecast.utils import normalize_item_name


def create_model_query(  # pylint: disable=too-many-arguments
    item_name: str,
    timestamp_col: str = "date",
    time_series_data_col: str = "total_amount_sold",
    model_name: str = "bqmlforecast.arima_plus_xreg_model",
    train_table_name: str = "bqmlforecast.training_data",
    test_table_name: str = "bqmlforecast.test_data",
    **kwargs,
) -> str:
    """
    Generate a BigQuery 'CREATE MODEL' query for a specified item.

    This function constructs an SQL query to create an ARIMA_PLUS_XREG
    model in BigQuery, tailored for the provided item and its associated
    data.

    Parameters
    ----------
    item_name : str
        Name of the item for which the model is to be created.
    timestamp_col : str, default="date"
        The column name representing the timestamp in the dataset.
    time_series_data_col : str, default="total_amount_sold"
        The column name representing the time series data.
    model_name : str, default="bqmlforecast.arima_plus_xreg_model"
        The base name for the model.
    train_table_name : str, default="bqmlforecast.training_data"
        The name of the table containing training data.
    test_table_name : str | None, default="bqmlforecast.test_data"
        The name of the table containing test data.
    **kwargs : Any
        Additional keyword arguments such as:

            holiday_region : str, default="US"
                The holiday region to be used by the model.
            auto_arima : bool, default=True
                Whether to enable AUTO_ARIMA.
            adjust_step_changes : bool, default=True
                Whether to adjust for step changes in the data.
            clean_spikes_and_dips : bool, default=True
                Whether to clean spikes and dips in the data.

    Returns
    -------
    str
        A SQL query string for creating the specified model.
    """
    configs = ARIMA_PLUS_XREG_Config(**kwargs)
    item_name_norm = normalize_item_name(item_name)
    test_table_query = include_test_on_model_train(item_name, timestamp_col,
                                                   train_table_name, test_table_name)
    return f"""
    CREATE OR REPLACE MODEL `{model_name}_{item_name_norm}`
    OPTIONS(
      MODEL_TYPE='ARIMA_PLUS_XREG',
      TIME_SERIES_TIMESTAMP_COL='{timestamp_col}',
      TIME_SERIES_DATA_COL='{time_series_data_col}',
      HOLIDAY_REGION='{configs.holiday_region}',
      AUTO_ARIMA={configs.auto_arima},
      ADJUST_STEP_CHANGES={configs.adjust_step_changes},
      CLEAN_SPIKES_AND_DIPS={configs.clean_spikes_and_dips}
    ) AS
    SELECT
        *
    FROM
        `{train_table_name}`
    WHERE
        item_name = "{item_name}"
    {test_table_query}
    ORDER BY
        date
    """


def include_test_on_model_train(
    item_name: str,
    timestamp_col: str,
    train_table_name: str,
    test_table_name: str | None = None,
) -> str:
    """
    Include test data in the model training process.

    This function generates an SQL query component to union test data with
    training data if a test table is specified.

    Parameters
    ----------
    item_name : str
        The name of the item being modeled.
    timestamp_col : str
        The column name representing the timestamp in the dataset.
    train_table_name : str
        The name of the table containing training data.
    test_table_name : str or None, optional
        The name of the table containing test data. If None, no test data
        is included.

    Returns
    -------
    str
        An SQL query string component to include test data.
    """
    if not isinstance(test_table_name, str):
        return ""
    return f"""
    UNION ALL
        (
            SELECT
                *
            FROM (
                SELECT
                    t2.*
                FROM
                    `{test_table_name}` AS  t2
                JOIN
                    (
                        SELECT
                            item_name,
                            MAX({timestamp_col}) AS max_date
                        FROM
                            `{train_table_name}`
                        GROUP BY
                            item_name
                    ) AS md
                ON
                    t2.item_name = md.item_name
                WHERE
                    t2.{timestamp_col} > md.max_date
                    AND t2.item_name = "{item_name}"
            )
        )
    """


def include_test_on_arima_model_train(
    column: str,
    time_series_timestamp_col: str,
    time_series_id_col: str,
    train_table_name: str,
    test_table_name: str | None = None,
) -> str:
    """
    Include test data in the uni-variate ARIMA model training process.

    This function generates an SQL query component to union test data with
    training data if a test table is specified.

    Parameters
    ----------
    column : str
        The name of the feature being modeled.
    time_series_timestamp_col : str
        The column name representing the timestamp in the dataset.
    time_series_id_col : str
        The column name representing the identifier.
    train_table_name : str
        The name of the table containing training data.
    test_table_name : str or None, optional
        The name of the table containing test data. If None, no test data
        is included.

    Returns
    -------
    str
        An SQL query string component to include test data.
    """
    if not isinstance(test_table_name, str):
        return ""
    return f"""
    UNION ALL
    (
        SELECT
            *
        FROM (
            SELECT
                t2.{time_series_timestamp_col},
                t2.{column},
                t2.{time_series_id_col}
            FROM
                `{test_table_name}` AS  t2
            JOIN
                (
                    SELECT
                        {time_series_id_col},
                        MAX({time_series_timestamp_col}) AS max_date
                    FROM
                        `{train_table_name}`
                    GROUP BY
                        {time_series_id_col}
                ) AS md
            ON
                t2.{time_series_id_col} = md.{time_series_id_col}
            WHERE
                t2.{time_series_timestamp_col} > md.max_date
        )
    )
    """


def execute_query_with_retries(
    client: bigquery.Client,
    query: str,
    max_retries: int = 3,
) -> None:
    """
    Execute a BigQuery SQL query with retries on failure.

    This function executes a given SQL query using a BigQuery client.
    If the query fails, it will automatically retry up to `max_retries`
    times, with an increasing delay between each attempt.

    Parameters
    ----------
    client : bigquery.Client
        Instance of the BigQuery client used to execute the query.
    query : str
        The SQL query to be executed.
    max_retries : int, default=3
        Maximum number of retry attempts in case of query failure.

    Raises
    ------
    Exception
        Raises an exception if all retry attempts fail.

    Notes
    -----
    The delay between retries increases linearly by 120 seconds
    multiplied by the current attempt number.

    Examples
    --------
    Execute a query with the default number of retries:

    >>> client = bigquery.Client()
    >>> query = "SELECT * FROM `my_dataset.my_table`"
    >>> execute_query_with_retries(client, query)
    """
    tries = 0
    success = False
    while not success and tries < max_retries:
        try:
            query_job = client.query(query)
            query_job.result()
            success = True
        except Exception as exc:  # pylint: disable=broad-except
            tries += 1
            sleep_time = 120 * tries
            print(exc)
            print(f"Attempt {tries} failed. Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)


def create_models_for_items(  # pylint: disable=too-many-arguments
    client: bigquery.Client,
    items_list: List[str],
    max_items: int | None = None,
    timestamp_col: str = "date",
    time_series_data_col: str = "total_amount_sold",
    model_name: str = "bqmlforecast.arima_plus_xreg_model",
    train_table_name: str = "bqmlforecast.training_data",
    test_table_name: str | None = "bqmlforecast.test_data",
    **kwargs,
) -> None:
    """
    Create `'ARIMA_PLUS_XREG'` models for a list of items.

    This function generates and executes a `'CREATE MODEL'` query
    for each item in the provided list. The models are created
    using the specified training and test tables in BigQuery.

    Parameters
    ----------
    client : bigquery.Client
        Instance of the BigQuery client used to execute queries.
    items_list : List[str]
        List of item names for which models are to be created.
    max_items : int or None, default=None
        Maximum number of items to process. If None, all items are processed.
        See the 'Notes' section for more information.
    timestamp_col : str, default="date"
        The column name representing the timestamp in the dataset.
    time_series_data_col : str, default="total_amount_sold"
        The column name representing the time series data.
    model_name : str, default="bqmlforecast.arima_plus_xreg_model"
        The base name for the models.
    train_table_name : str, default="bqmlforecast.training_data"
        The name of the table containing training data.
    test_table_name : str | None, default="bqmlforecast.test_data"
        The name of the table containing test data.
        If `None`, then only the data from `train_table_name` is used for
        training the model. See the 'Notes' section for more information.
    **kwargs : Any
        Additional keyword arguments such as:

            holiday_region : str, default="US"
                The holiday region to be used by the models.
            auto_arima : bool, default=True
                Whether to enable `'AUTO_ARIMA'`.
            adjust_step_changes : bool, default=True
                Whether to adjust for step changes in the data.
            clean_spikes_and_dips : bool, default=True
                Whether to clean spikes and dips in the data.

    Notes
    -----
    Not specifying a value for `max_items` requires you to use a Google Cloud
    account with billing enabled. If you're not using a Google Cloud account
    with billing enabled, then you should limit the number of items
    to a value smaller than or equal to 4.

    .. important::

        If using a Google Cloud account with billing enabled, running this
        code might incur charges.

    If you are evaluating the model, you shouldn't use all available data
    to train the model. Therefore, if you're evaluating the model, consider
    setting the parameter `test_table_name` to `None`. Doing so will cause
    the model to be trained using only the specified data from the
    `train_table_name` which in turn will allow you to use the data from
    `test_table_name` for evaluation.
    """
    _items_list = (
        items_list if not isinstance(max_items, int) else items_list[:max_items]
    )
    for item_name in track(_items_list, description="Creating models..."):
        query = create_model_query(
            item_name,
            timestamp_col,
            time_series_data_col,
            model_name,
            train_table_name,
            test_table_name,
            **kwargs,
        )
        execute_query_with_retries(client, query)


def train_arima_models(  # pylint: disable=too-many-locals, too-many-arguments
    client: bigquery.Client,
    columns: List[str],
    model: str = "bqmlforecast.arima_model",
    train_table_name: str = "bqmlforecast.training_data",
    test_table_name: str | None = "bqmlforecast.test_data",
    model_metrics_table_name: str | None = "bqmlforecast.arima_model_metrics",
    time_series_timestamp_col: str = "date",
    time_series_id_col: str = "item_name",
    confidence_level: float = 0.9,
    horizon: int = 7,
    use_test_data_on_train: bool = True,
    **kwargs,
):
    """
    Train ARIMA models for a list of columns and store their metrics.

    This function generates and executes `'CREATE MODEL'` queries for ARIMA
    models using the specified columns, and evaluates their performance
    by creating tables of model metrics.

    These ARIMA models will then be used to generate the future feature values
    used for forecasting the liquor sales.

    Parameters
    ----------
    client : bigquery.Client
        Instance of the BigQuery client used to execute queries.
    columns : List[str]
        List of columns to be used for creating ARIMA models.
    model : str, default="bqmlforecast.arima_model"
        The base name for the ARIMA models.
    train_table_name : str, default="bqmlforecast.training_data"
        The name of the table containing training data.
    test_table_name : str | None, default="bqmlforecast.test_data"
        The name of the table containing test data.
    model_metrics_table_name : str or None, default="bqmlforecast.arima_model_metrics"
        The base name for the tables where model metrics will be stored.
    time_series_timestamp_col : str, default="date"
        The column name representing the timestamp in the dataset.
    time_series_id_col : str, default="item_name"
        The column name representing the identifier for the time series.
    confidence_level : float, default=0.9
        The confidence level used in the model evaluation.
    horizon : int, default=7
        The number of time steps (days) to forecast.
    use_test_data_on_train : bool, default=True
        Whether to use test data during model training.
    """
    config = ARIMAConfig(**kwargs)

    for column in track(columns, description="Creating ARIMA models..."):
        model_name = f"{model}_{column}"
        test_data_query = ""
        if use_test_data_on_train:
            test_data_query = include_test_on_arima_model_train(
                column,
                time_series_timestamp_col,
                time_series_id_col,
                train_table_name,
                test_table_name,
            )
        train_arima_query = f"""
        CREATE OR REPLACE MODEL `{model_name}`
        OPTIONS(
            MODEL_TYPE = '{config.model_type}',
            AUTO_ARIMA = {config.auto_arima},
            HORIZON = {horizon},
            TIME_SERIES_TIMESTAMP_COL = '{time_series_timestamp_col}',
            TIME_SERIES_DATA_COL = '{column}',
            TIME_SERIES_ID_COL = '{time_series_id_col}',
            FORECAST_LIMIT_LOWER_BOUND = {config.forecast_limit_lower_bound},
            DECOMPOSE_TIME_SERIES = {config.decompose_time_series},
            HOLIDAY_REGION = '{config.holiday_region}',
            DATA_FREQUENCY = '{config.data_frequency}',
            ADJUST_STEP_CHANGES = {config.adjust_step_changes},
            CLEAN_SPIKES_AND_DIPS = {config.clean_spikes_and_dips}
        ) AS
        SELECT
            {time_series_timestamp_col},
            {column},
            {time_series_id_col}
        FROM
            `{train_table_name}`
        {test_data_query}
        """
        train_arima_job = client.query(train_arima_query)
        train_arima_job.result()

        if isinstance(model_metrics_table_name, str):
            model_metrics_query = f"""
            CREATE OR REPLACE TABLE `{model_metrics_table_name}_{column}` AS (
            SELECT
                *
            FROM
                ML.EVALUATE(
                    MODEL `{model_name}`,
                    (
                        SELECT
                            {time_series_timestamp_col},
                            {time_series_id_col},
                            {column}
                        FROM
                            `{test_table_name}`
                    ),
                    STRUCT({horizon} AS HORIZON, {confidence_level} AS CONFIDENCE_LEVEL)
                )
            )
            """
            model_metrics_job = client.query(model_metrics_query)
            model_metrics_job.result()
