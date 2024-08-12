"""
BigQuery Data Loading and Feature Engineering Module.

This module provides functions to load, process, and prepare data for
time series forecasting using Google BigQuery. The module includes
utilities for creating training datasets, generating forecast features,
and handling date offsets and item filters.

Functions
---------
* `date_offset`: generate a pandas DateOffset based on the given frequency and value.

* `get_item_names_filter`: generate a "WHERE" clause component to filter values
  from column `"item_name"`.

* `get_min_datapoints_filter`: generate a "WHERE" clause to filter items that
  have at least `min_size` observations.

* `get_training_data`: retrieve data from BigQuery and create a training data view.

* `get_year_weather_query`: generate an SQL query to retrieve weather data for
  a specific year and state.

* `get_weather_query`: generate an SQL query to retrieve weather data for
  a given date range.

* `create_forecast_features_query`: generate an SQL query to create forecast
  features by joining forecast tables.

* `create_future_data`: create future data features using training, test,
  and forecast data.

* `create_future_feature_table`: create a forecast table for a specific model
  and save the forecast results.

* `create_future_feature_tables`: create forecast tables for multiple columns
  and save the forecast results.

Notes
-----
This module is designed to work with Google BigQuery and requires a valid
BigQuery client instance. The functions in this module focus on preparing
data for time series forecasting in various business contexts.

See Also
--------
Google BigQuery: https://cloud.google.com/bigquery
BigQuery ML: https://cloud.google.com/bigquery-ml
"""
from __future__ import annotations

from typing import List

import google
import pandas as pd
from google.cloud import bigquery
from rich.progress import track

from iowa_forecast.utils import list_tables_with_pattern, date_offset


def get_item_names_filter(items_list: List[str] | str) -> str:
    """
    Generate a `"WHERE"` clause component to filter values from column `"item_name"`.

    Parameters
    ----------
    items_list : List[str] | str
        Item name or names to add to the `"WHERE"` clause component.

    Returns
    -------
    str
        The `"WHERE"` clause component that can be used to filter values from column `"item_name"`.

    Examples
    --------
    >>> print(get_item_names_filter("FIVE O'CLOCK VODKA"))
    (item_name = "FIVE O'CLOCK VODKA")
    >>> print(get_item_names_filter(['FIREBALL CINNAMON WHISKEY', 'BLACK VELVET']))
    (item_name = "FIREBALL CINNAMON WHISKEY" OR item_name = "BLACK VELVET")

    """
    if isinstance(items_list, str):
        items_list = [items_list]

    return "(" + " OR ".join(f'item_name = "{item_name}"' for item_name in items_list) + ")"


def get_min_datapoints_filter(min_size: int) -> str:
    """
    Generate a `"WHERE"` clause to filter items that have at least `min_size` observations.

    Parameters
    ----------
    min_size : int
        Minimum number of observations to use as value for the `"WHERE"` clause.

    Returns
    -------
    str
        The `"WHERE"` clause component.
    """
    return f"""
    WHERE
        example_count >= {min_size}
        AND DATE_DIFF(last_sale_date, first_sale_date, DAY) >= {min_size}
    """


def get_training_data(
    client: bigquery.Client,
    table_name: str = 'bqmlforecast.training_data',
    start_date: str | None = None,
    end_date: str | None = None,
    offset: int | None = None,
    freq: str = "years",
    min_datapoints_rate: float = 0.75,
    items_list: str | List[str] | None = None,
    base_table: str = "bigquery-public-data.iowa_liquor_sales.sales",
) -> pd.DataFrame:
    """
    Retrieve data from BigQuery and create a training data view.

    This function constructs an SQL query to create a view in BigQuery,
    filters sales data based on the specified date range and conditions,
    and retrieves the filtered data as a pandas DataFrame.

    Parameters
    ----------
    client : bigquery.Client
        BigQuery client object.
    table_name : str, default='bqmlforecast.training_data'
        The name of the table to store the training data view.
    start_date : str, optional
        The start date for filtering data in 'YYYY-MM-DD' format.
        If `None`, the start date is determined by one of the following ways:
    end_date : str, optional
        The end date for filtering data in 'YYYY-MM-DD' format.
        If `None`, the end date is determined by one of the following ways:

            * If `offset` is not `None`: then `end_date` equals `start_date` + `offset`
            * If `offset` is `None`: then `end_date` is today's date.

    offset : int, optional
        The offset value to calculate the start or end date.
    freq : str, default='years'
        The frequency type for the offset calculation.
    min_datapoints_rate : float, default=0.75
        The fraction of days between `end_date` and `start_date` that each
        item from 'item_name' column should have of data points to be considered
        in the created table.
    items_list : str | List[str], optional
        A list of item names or single item name used for filtering the data.
    base_table : str, default="bigquery-public-data.iowa_liquor_sales.sales"
        Base table to extract data from.

    Returns
    -------
    pd.DataFrame
        A `pandas.DataFrame` containing the filtered sales data.

    Raises
    ------
    ValueError
        If none of `start_date`, `end_date`, or `offset` are specified.
        If `start_date` > `end_date`.
    """
    if not any([start_date, end_date, offset]):
        raise ValueError(
            "At least one of the following parameters must be specified: "
            "`start_date`, `end_date` and `offset`."
        )

    if end_date is None:
        if offset is None:
            end_date = pd.Timestamp.today()
        else:
            end_date = pd.Timestamp(start_date) + date_offset(offset, freq)

    elif start_date is None:
        if offset is None:
            raise ValueError(
                "If start_date is None, then `offset` and "
                "`end_date` must be specified."
            )
        start_date = pd.Timestamp(end_date) - date_offset(offset, freq)

    start_date_ts = pd.Timestamp(start_date)
    end_date_ts = pd.Timestamp(end_date)

    start_date = start_date_ts.strftime("%Y-%m-%d")
    end_date = end_date_ts.strftime("%Y-%m-%d")

    days_range = pd.date_range(start_date, end_date, freq="D")
    where_min_datapoints_query = ""
    if isinstance(min_datapoints_rate, float):
        min_data_points = int(min_datapoints_rate * len(days_range))
        where_min_datapoints_query = get_min_datapoints_filter(min_data_points)

    if start_date_ts > end_date_ts:
        raise ValueError(f"`start_date` > `end_date`: {start_date} > {end_date}")

    weather_query = get_weather_query(start_date, end_date)

    filter_items_query = ""
    if items_list is not None:
        filter_items_query += f" AND {get_item_names_filter(items_list)}"

    query = f"""
    CREATE OR REPLACE TABLE `{table_name}` AS (
    WITH date_range AS (
        SELECT
            date
        FROM
            UNNEST(GENERATE_DATE_ARRAY(DATE_SUB(DATE('{start_date}'), INTERVAL 60 DAY), '{end_date}', INTERVAL 1 DAY)) AS date
    ),
    sales_data AS (
        SELECT
            date,
            item_description AS item_name,
            SUM(bottles_sold) AS total_amount_sold,
            AVG(state_bottle_retail) AS avg_bottle_price,
            AVG(state_bottle_cost) AS avg_bottle_cost,
            SUM(volume_sold_liters) AS total_volume_sold_liters,
        FROM
            `{base_table}`
        WHERE
            bottles_sold > 0
            AND sale_dollars > 0
            AND city IS NOT NULL
        GROUP BY
            date, item_name
        HAVING
            date >= DATE_SUB(DATE('{start_date}'), INTERVAL 60 DAY)
    ),
    weather_data AS (
        {weather_query}
    ),
    filtered_sales AS (
        SELECT
            dr.date,
            sd.item_name,
            COALESCE(sd.total_amount_sold, 0) AS total_amount_sold,
            COALESCE(sd.avg_bottle_price, 0) AS avg_bottle_price,
            COALESCE(sd.avg_bottle_cost, 0) AS avg_bottle_cost,
            COALESCE(sd.total_volume_sold_liters, 0) AS total_volume_sold_liters,
            COALESCE(wd.temperature, 0) AS temperature,
            COALESCE(wd.rainfall, 0) AS rainfall,
            COALESCE(wd.snowfall, 0) AS snowfall
        FROM
            date_range dr
        LEFT JOIN sales_data sd ON dr.date = sd.date
        LEFT JOIN weather_data wd ON dr.date = wd.date
    ),
    item_stats AS (
        SELECT
            item_name,
            COUNT(*) AS example_count,
            COUNTIF(total_amount_sold > 0) AS sales_days_count,
            MIN(date) AS first_sale_date,
            MAX(date) AS last_sale_date
        FROM
            filtered_sales
        GROUP BY
            item_name
    ),
    filtered_items AS (
        SELECT
            item_name
        FROM
            item_stats
        {where_min_datapoints_query}
    ),
    filtered_sales_final AS (
        SELECT
            fs.*
        FROM
            filtered_sales fs
        JOIN
            filtered_items fi ON fs.item_name = fi.item_name
    ),
    lag_features AS (
        SELECT
            *,
            LAG(COALESCE(total_volume_sold_liters, 0), 1) OVER (PARTITION BY item_name ORDER BY date) AS lag_1_total_volume_sold_liters,
            LAG(COALESCE(total_volume_sold_liters, 0), 7) OVER (PARTITION BY item_name ORDER BY date) AS lag_7_total_volume_sold_liters,
        FROM
            filtered_sales_final
    ),
    final_features AS (
        SELECT
            *,
            AVG(COALESCE(total_volume_sold_liters, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ma3_total_volume_sold_liters,
            AVG(COALESCE(avg_bottle_price, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ma3_avg_bottle_price,
            AVG(COALESCE(total_volume_sold_liters, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7_total_volume_sold_liters,
            AVG(COALESCE(avg_bottle_price, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7_avg_bottle_price,
            AVG(COALESCE(total_volume_sold_liters, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS ma30_total_volume_sold_liters,
            AVG(COALESCE(avg_bottle_price, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS ma30_avg_bottle_price,
        FROM
            lag_features
    )
    SELECT
        date,
        item_name,
        ROUND(total_amount_sold, 0) AS total_amount_sold,
        ROUND(avg_bottle_price, 2) AS avg_bottle_price,
        ROUND(total_volume_sold_liters, 2) AS total_volume_sold_liters,
        ROUND(avg_bottle_cost, 2) AS avg_bottle_cost,
        EXTRACT(DAYOFWEEK FROM date) AS day_of_week,
        EXTRACT(WEEK FROM date) AS week_of_year,
        EXTRACT(MONTH FROM date) AS month,
        EXTRACT(YEAR FROM date) AS year,
        ROUND(ma3_total_volume_sold_liters, 2) AS ma3_total_volume_sold_liters,
        ROUND(ma3_avg_bottle_price, 2) AS ma3_avg_bottle_price,
        ROUND(ma7_total_volume_sold_liters, 2) AS ma7_total_volume_sold_liters,
        ROUND(ma7_avg_bottle_price, 2) AS ma7_avg_bottle_price,
        ROUND(ma30_total_volume_sold_liters, 2) AS ma30_total_volume_sold_liters,
        ROUND(ma30_avg_bottle_price, 2) AS ma30_avg_bottle_price,
        ROUND(temperature, 2) AS temperature,
        (CASE WHEN rainfall IS NULL THEN 0 ELSE ROUND(rainfall, 2) END) AS rainfall,
        (CASE WHEN snowfall IS NULL THEN 0 ELSE ROUND(snowfall, 2) END) AS snowfall,
        lag_1_total_volume_sold_liters,
        lag_7_total_volume_sold_liters,
    FROM
        final_features
    WHERE
        date BETWEEN DATE('{start_date}') AND DATE('{end_date}')
        AND item_name IS NOT NULL
        AND lag_7_total_volume_sold_liters IS NOT NULL
        AND lag_1_total_volume_sold_liters IS NOT NULL
        {filter_items_query}
    ORDER BY
        item_name, date
    )
    """
    query_job = client.query(query)
    print(query_job.result())
    return client.query(f"SELECT * FROM `{table_name}`").to_dataframe()


def get_year_weather_query(year: int, state: str = "IA") -> str:
    """
    Generate an SQL query to retrieve weather data for a specific year and state.

    Parameters
    ----------
    year : int
        The year for which to retrieve weather data.
    state : str, default="IA"
        The state code for which to retrieve weather data.

    Returns
    -------
    str
        SQL query string to retrieve the weather data.
    """
    return f"""
    SELECT
        date,
        AVG(temp) AS temperature,
        AVG(CASE WHEN prcp = 999.9 THEN 0 ELSE prcp END) AS rainfall,
        AVG(CASE WHEN sndp = 999.9 THEN 0 ELSE sndp END) AS snowfall
    FROM
        `bigquery-public-data.noaa_gsod.gsod{year}`
    WHERE
        stn IN (SELECT usaf FROM `bigquery-public-data.noaa_gsod.stations` WHERE state = '{state}')
    GROUP BY
      date
    """


def get_weather_query(start_date: str, end_date: str, state: str = "IA") -> str:
    """
    Generate an SQL query to retrieve weather data for a given date range.

    Parameters
    ----------
    start_date : str
        The start date for the weather data in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the weather data in 'YYYY-MM-DD' format.
    state : str, default="IA"
        The state code for which to retrieve weather data.

    Returns
    -------
    str
        SQL query string to retrieve the weather data.
    """
    years = {date.year for date in pd.date_range(start_date, end_date, freq="D")}
    weather_year_queries = [get_year_weather_query(year, state) for year in years]
    return " UNION ALL ".join(weather_year_queries)


def create_forecast_features_query(
    client: bigquery.Client,
    dataset_id: str = "bqmlforecast",
    forecast_tables_pattern: str = "forecast_*",
) -> str:
    """
    Generate an SQL query to create forecast features by joining forecast tables.

    Parameters
    ----------
    client : bigquery.Client
        BigQuery client used to connect to the service.
    dataset_id : str, optional
        ID of the dataset where the forecast tables are located,
        by default "bqmlforecast".
    forecast_tables_pattern : str, optional
        Pattern to match forecast tables, by default "forecast_*".

    Returns
    -------
    str
        SQL query string to create forecast features.

    Examples
    --------
    >>> client = bigquery.Client()
    >>> query = create_forecast_features_query(client)
    >>> print(query)
    SELECT
        CAST(t1.forecast_timestamp AS DATE) as date,
        t1.item_name,
        0 AS total_amount_sold,
        t2.forecast_value AS temp,
        t3.forecast_value AS rainfall,
        t4.forecast_value AS snowfall,
        fw.rainfall,
        fw.snowfall,
        fw.temperature
    FROM
        `forecast_temp` AS t1
    INNER JOIN `forecast_rainfall` AS t2
        ON t1.forecast_timestamp = t2.forecast_timestamp
        AND t1.item_name = t2.item_name
    INNER JOIN `forecast_snowfall` AS t3
        ON t2.forecast_timestamp = t3.forecast_timestamp
        AND t2.item_name = t3.item_name
    LEFT JOIN future_weather_data AS fw
        ON date = fw.date

    Notes
    -----
    This function constructs an SQL query by joining multiple forecast tables
    and a weather data table to create a comprehensive forecast features dataset.
    """
    project_id = client.project
    table_names = list_tables_with_pattern(client, dataset_id, forecast_tables_pattern)
    forecast_columns_pattern = forecast_tables_pattern.replace("*", "")
    join_clauses = []
    select_clauses = [
        "CAST(t1.forecast_timestamp AS DATE) as date",
        "t1.item_name",
        "0 AS total_amount_sold"
    ]

    for idx, table in enumerate(table_names):
        alias = f"t{idx + 1}"
        columns_name = table.replace(forecast_columns_pattern, "")
        select_clauses.append(f"{alias}.forecast_value AS {columns_name}")
        if idx > 0:
            previous_alias = f"t{idx}"
            join_clauses.append(
                f"INNER JOIN `{project_id}.{dataset_id}.{table}` AS {alias} "
                f"ON {previous_alias}.forecast_timestamp = {alias}.forecast_timestamp "
                f"AND {previous_alias}.item_name = {alias}.item_name"
            )

    select_clauses.extend([
        "fw.rainfall",
        "fw.snowfall",
        "fw.temperature"
    ])

    select_clause_str = ",\n".join(select_clauses)
    join_clause_str = "\n".join(join_clauses)

    query = f"""
    SELECT
        {select_clause_str}
    FROM
        `{project_id}.{dataset_id}.{table_names[0]}` AS t1
    {join_clause_str}
    LEFT JOIN
        future_weather_data AS fw
    ON
        date = fw.date
    """

    return query


def create_future_data(
    client: bigquery.Client,
    train_table_name: str = "bqmlforecast.training_data",
    test_table_name: str = "bqmlforecast.test_data",
    forecast_table_name: str = "bqmlforecast.forecast_data",
    horizon: int = 7,
    state: str = "IA",
    dataset_id: str = "bqmlforecast",
):
    """
    Create future data features using training, test, and forecast data.

    This function generates an SQL query to create a future dataset
    based on the training, test, and forecast data. It includes
    weather information and lag features.

    Parameters
    ----------
    client : bigquery.Client
        BigQuery client used to connect to the service.
    train_table_name : str, default="bqmlforecast.training_data"
        The name of the training data table.
    test_table_name : str, default="bqmlforecast.test_data"
        The name of the test data table.
    forecast_table_name : str, default="bqmlforecast.forecast_data"
        The name of the table to store the forecast data.
    horizon : int, default=7
        The number of days into the future to create data for.
    state : str, default="IA"
        The state code for weather data retrieval.
    dataset_id : str, default="bqmlforecast"
        The dataset ID where the forecast tables are located.
    """
    xdf = client.query(f"SELECT date FROM `{test_table_name}`").to_dataframe()
    end_date = xdf["date"].max()
    end_date = pd.Timestamp(end_date)
    previous_year_date = end_date - pd.DateOffset(years=1)
    previous_year = int(previous_year_date.year)
    end_date = end_date.strftime("%Y-%m-%d")

    forecast_features_query = create_forecast_features_query(client, dataset_id,
                                                             "forecast_*")

    future_data_query = f"""
    CREATE OR REPLACE TABLE `{forecast_table_name}` AS (
    WITH max_date AS (
        SELECT MAX(date) AS max_date
        FROM `{test_table_name}`
    ),
    future_dates AS (
        SELECT
            DATE_ADD(max_date.max_date, INTERVAL day_offset DAY) AS date
        FROM
            max_date,
            UNNEST(GENERATE_ARRAY(1, {horizon})) AS day_offset
    ),
    future_weather_data AS (
        SELECT
            fd.date,
            AVG(hd.temp) AS temperature,
            AVG(CASE WHEN hd.prcp = 999.9 THEN 0 ELSE hd.prcp END) AS rainfall,
            AVG(CASE WHEN hd.sndp = 999.9 THEN 0 ELSE hd.sndp END) AS snowfall
        FROM
            `future_dates` fd
        LEFT JOIN
            `bigquery-public-data.noaa_gsod.gsod{previous_year}` hd
        ON
            DATE_ADD(hd.date, INTERVAL 1 YEAR) = fd.date
        WHERE
            stn IN (SELECT usaf FROM `bigquery-public-data.noaa_gsod.stations` WHERE state = '{state}')
        GROUP BY
            fd.date
    ),
    forecast_features AS (
        {forecast_features_query}
    ),
    validation_data AS (
        SELECT
            date,
            item_name,
            total_amount_sold,
            avg_bottle_cost,
            avg_bottle_price,
            total_volume_sold_liters,
            rainfall,
            snowfall,
            temperature
        FROM
            `{test_table_name}`
    ),
    training_data AS (
        SELECT
            date,
            item_name,
            total_amount_sold,
            avg_bottle_cost,
            avg_bottle_price,
            total_volume_sold_liters,
            rainfall,
            snowfall,
            temperature
        FROM
            `{train_table_name}`
        UNION ALL
            SELECT * FROM validation_data
        UNION ALL
            SELECT * FROM forecast_features
    ),
    lag_features AS (
        SELECT
            *,
            LAG(COALESCE(total_volume_sold_liters, 0), 1) OVER (PARTITION BY item_name ORDER BY date) AS lag_1_total_volume_sold_liters,
            LAG(COALESCE(total_volume_sold_liters, 0), 7) OVER (PARTITION BY item_name ORDER BY date) AS lag_7_total_volume_sold_liters
        FROM
            training_data
            WHERE
                total_amount_sold IS NOT NULL
                AND avg_bottle_price IS NOT NULL
                AND avg_bottle_cost IS NOT NULL
                AND total_volume_sold_liters IS NOT NULL
    ),
    final_features AS (
        SELECT
            *,
            AVG(COALESCE(total_volume_sold_liters, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ma3_total_volume_sold_liters,
            AVG(COALESCE(avg_bottle_price, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ma3_avg_bottle_price,
            AVG(COALESCE(total_volume_sold_liters, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7_total_volume_sold_liters,
            AVG(COALESCE(avg_bottle_price, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7_avg_bottle_price,
            AVG(COALESCE(total_volume_sold_liters, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS ma30_total_volume_sold_liters,
            AVG(COALESCE(avg_bottle_price, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS ma30_avg_bottle_price
        FROM
            lag_features
    ),
    deduplicated_final_features AS (
        SELECT
            *,
            ROW_NUMBER() OVER (PARTITION BY date, item_name ORDER BY date) AS row_num
        FROM
            final_features
    )
    SELECT
        date,
        item_name,
        ROUND(total_amount_sold, 0) AS total_amount_sold,
        ROUND(avg_bottle_price, 2) AS avg_bottle_price,
        ROUND(total_volume_sold_liters, 2) AS total_volume_sold_liters,
        ROUND(avg_bottle_cost, 2) AS avg_bottle_cost,
        EXTRACT(DAYOFWEEK FROM date) AS day_of_week,
        EXTRACT(WEEK FROM date) AS week_of_year,
        EXTRACT(MONTH FROM date) AS month,
        EXTRACT(YEAR FROM date) AS year,
        ROUND(ma3_total_volume_sold_liters, 2) AS ma3_total_volume_sold_liters,
        ROUND(ma3_avg_bottle_price, 2) AS ma3_avg_bottle_price,
        ROUND(ma7_total_volume_sold_liters, 2) AS ma7_total_volume_sold_liters,
        ROUND(ma7_avg_bottle_price, 2) AS ma7_avg_bottle_price,
        ROUND(ma30_total_volume_sold_liters, 2) AS ma30_total_volume_sold_liters,
        ROUND(ma30_avg_bottle_price, 2) AS ma30_avg_bottle_price,
        ROUND(LAST_VALUE(temperature IGNORE NULLS) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS temperature,
        (CASE WHEN rainfall IS NULL THEN 0 ELSE ROUND(rainfall, 2) END) AS rainfall,
        (CASE WHEN snowfall IS NULL THEN 0 ELSE ROUND(snowfall, 2) END) AS snowfall,
        CAST(lag_1_total_volume_sold_liters AS INT64) AS lag_1_total_volume_sold_liters,
        CAST(lag_7_total_volume_sold_liters AS INT64) AS lag_7_total_volume_sold_liters
    FROM
        deduplicated_final_features
    WHERE
        EXTRACT(DAYOFWEEK FROM date) IS NOT NULL
        AND EXTRACT(WEEK FROM date) IS NOT NULL
        AND EXTRACT(MONTH FROM date)  IS NOT NULL
        AND EXTRACT(YEAR FROM date) IS NOT NULL
        AND row_num = 1
    ORDER BY item_name, date
    )
    """
    future_data_job = client.query(future_data_query)
    future_data_job.result()


def create_future_feature_table(
    client: bigquery.Client,
    table_name: str,
    model_name: str,
    confidence_level: float = 0.9,
    horizon: int = 7,
):
    """
    Create a forecast table for a specific model and save the forecast results.

    Parameters
    ----------
    client : bigquery.Client
        BigQuery client used to connect to the service.
    table_name : str
        The name of the table to save the forecast results.
    model_name : str
        The name of the model used for forecasting.
    confidence_level : float, default=0.9
        The confidence level for the forecast.
    horizon : int, default=7
        The number of days into the future to forecast.
    """
    create_table_query = f"""
    CREATE OR REPLACE TABLE `{table_name}` AS (
        SELECT
         *
        FROM
         ML.FORECAST(MODEL `{model_name}`,
                     STRUCT({horizon} AS horizon, {confidence_level} AS confidence_level))
    )
    """
    create_table_job = client.query(create_table_query)
    create_table_job.result()


def create_future_feature_tables(
    client: bigquery.Client,
    columns: List[str],
    model: str = "bqmlforecast.arima_model",
    table_base_name: str = "bqmlforecast.forecast",
    confidence_level: float = 0.9,
    horizon: int = 7,
):
    """
    Create forecast tables for multiple columns and save the forecast results.

    Parameters
    ----------
    client : bigquery.Client
        BigQuery client used to connect to the service.
    columns : List[str]
        A list of column names to forecast.
    model : str, default="bqmlforecast.arima_model"
        The base name of the model used for forecasting.
    table_base_name : str, default="bqmlforecast.forecast"
        The base name for the tables to store the forecast results.
    confidence_level : float, default=0.9
        The confidence level for the forecast.
    horizon : int, default=7
        The number of days into the future to forecast.
    """
    for column in track(columns, description="Saving ARIMA forecasts..."):
        model_name = f"{model}_{column}"
        table_name = f"{table_base_name}_{column}"
        create_future_feature_table(
            client,
            table_name,
            model_name,
            confidence_level=confidence_level,
            horizon=horizon
        )
