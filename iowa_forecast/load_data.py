from __future__ import annotations

from typing import List

import google
import pandas as pd

from iowa_forecast.ml_eval import get_train_data


def date_offset(n: int, freq: str) -> pd.DateOffset:
    """Generate a pandas DateOffset based on the given frequency and value.

    Parameters
    ----------
    n : int
        The number of time units for the offset.
    freq : str {'days', 'weeks', 'months', 'years'}
        The frequency type. Valid options are 'days', 'weeks', 'months', 'years'.

    Returns
    -------
    pd.DateOffset
        A DateOffset object for the specified frequency and value.

    Raises
    ------
    ValueError
        If `freq` is not one of the valid options.
    """
    if freq == "days":
        return pd.DateOffset(days=n)
    if freq == "weeks":
        return pd.DateOffset(weeks=n)
    if freq == "months":
        return pd.DateOffset(months=n)
    if freq == "years":
        return pd.DateOffset(years=n)
    raise ValueError(
        f"The specified `freq` {freq} is not a valid frequency. "
        "Valid frequencies are: 'days', 'weeks', 'months', 'years'."
    )


def get_item_names_filter(items_list: List[str] | str) -> str:
    """
    Generate a "WHERE" clause component to filter values from column `"item_name"`.

    items_list : List[str] | str
        Item name or names to add to the "WHERE" clause component.

    Returns
    -------
    str
        The "WHERE" clause component that can be used to filter values from column `"item_name".

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
    Generate a "WHERE" clause to filter items that have at least `min_size` observations.

    min_size : int
        Minimum number of observations to use as value for the "WHERE" clause.

    Returns
    -------
    str
        The "WHERE" clause component.
    """
    return f"""
    WHERE
        example_count >= {min_size}
        AND DATE_DIFF(last_sale_date, first_sale_date, DAY) >= {min_size}
    """


def get_training_data(
    client: google.cloud.bigquery.Client,
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
    client : google.cloud.bigquery.Client
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
            LAG(COALESCE(total_amount_sold, 0), 1) OVER (PARTITION BY item_name ORDER BY date) AS lag_1_total_amount_sold,
            LAG(COALESCE(total_amount_sold, 0), 7) OVER (PARTITION BY item_name ORDER BY date) AS lag_7_total_amount_sold,
        FROM
            filtered_sales_final
    ),
    final_features AS (
        SELECT
            *,
            AVG(COALESCE(total_amount_sold, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ma3_total_amount_sold,
            AVG(COALESCE(avg_bottle_price, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ma3_avg_bottle_price,
            AVG(COALESCE(total_amount_sold, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7_total_amount_sold,
            AVG(COALESCE(avg_bottle_price, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7_avg_bottle_price,
            AVG(COALESCE(total_amount_sold, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS ma30_total_amount_sold,
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
        ROUND(ma3_total_amount_sold, 2) AS ma3_total_amount_sold,
        ROUND(ma3_avg_bottle_price, 2) AS ma3_avg_bottle_price,
        ROUND(ma7_total_amount_sold, 2) AS ma7_total_amount_sold,
        ROUND(ma7_avg_bottle_price, 2) AS ma7_avg_bottle_price,
        ROUND(ma30_total_amount_sold, 2) AS ma30_total_amount_sold,
        ROUND(ma30_avg_bottle_price, 2) AS ma30_avg_bottle_price,
        ROUND(temperature, 2) AS temperature,
        (CASE WHEN rainfall IS NULL THEN 0 ELSE ROUND(rainfall, 2) END) AS rainfall,
        (CASE WHEN snowfall IS NULL THEN 0 ELSE ROUND(snowfall, 2) END) AS snowfall,
        lag_1_total_amount_sold,
        lag_7_total_amount_sold,
    FROM
        final_features
    WHERE
        date BETWEEN DATE('{start_date}') AND DATE('{end_date}')
        AND item_name IS NOT NULL
        AND lag_7_total_amount_sold IS NOT NULL
        AND lag_1_total_amount_sold IS NOT NULL
        {filter_items_query}
    ORDER BY
        item_name, date
    )
    """
    query_job = client.query(query)
    print(query_job.result())
    return client.query(f"SELECT * FROM `{table_name}`").to_dataframe()


def get_year_weather_query(year: int, state: str = "IA") -> str:
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
    years = {date.year for date in pd.date_range(start_date, end_date, freq="D")}
    weather_year_queries = [get_year_weather_query(year, state) for year in years]
    return " UNION ALL ".join(weather_year_queries)


def create_future_data_query(
    client,
    train_table_name: str = "bqmlforecast.training_data",
    forecast_table_name: str = "bqmlforecast.forecast_data",
    horizon: int = 7,
    state: str = "IA"
):
    xdf = client.query(f"SELECT date FROM `{train_table_name}`").to_dataframe()
    end_date = xdf["date"].max()
    end_date = pd.Timestamp(end_date)
    previous_year_date = end_date - pd.DateOffset(years=1)
    previous_year = int(previous_year_date.year)
    end_date = end_date.strftime("%Y-%m-%d")

    future_data_query = f"""
    CREATE OR REPLACE TABLE `{forecast_table_name}` AS (
    WITH max_date AS (
        SELECT MAX(date) AS max_date
        FROM `{train_table_name}`
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
        SELECT
            CAST(t1.forecast_timestamp AS DATE) as date,
            t1.item_name,
            0 AS total_amount_sold,
            t1.forecast_value AS avg_bottle_price,
            t2.forecast_value AS avg_bottle_cost,
            t3.forecast_value AS total_volume_sold_liters,
            fw.rainfall,
            fw.snowfall,
            fw.temperature
        FROM
            `bqmlforecast.forecast_avg_bottle_price` AS t1
        INNER JOIN
            `bqmlforecast.forecast_avg_bottle_cost` AS t2
        ON
            t1.forecast_timestamp = t2.forecast_timestamp AND t1.item_name = t2.item_name
        INNER JOIN
            `bqmlforecast.forecast_total_volume_sold_liters` AS t3
        ON
            t1.forecast_timestamp = t3.forecast_timestamp AND t1.item_name = t3.item_name
        LEFT JOIN
            future_weather_data AS fw
        ON
            date = fw.date
    ),
    training_data AS (
        SELECT
            date,
            item_name,
            total_amount_sold,
            avg_bottle_price,
            avg_bottle_cost,
            total_volume_sold_liters,
            rainfall,
            snowfall,
            temperature
        FROM
            `bqmlforecast.training_data`
        UNION ALL
            SELECT * FROM forecast_features
    ),
    lag_features AS (
        SELECT
            *,
            LAG(COALESCE(total_amount_sold, 0), 1) OVER (PARTITION BY item_name ORDER BY date) AS lag_1_total_amount_sold,
            LAG(COALESCE(total_amount_sold, 0), 7) OVER (PARTITION BY item_name ORDER BY date) AS lag_7_total_amount_sold
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
            AVG(COALESCE(total_amount_sold, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ma3_total_amount_sold,
            AVG(COALESCE(avg_bottle_price, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ma3_avg_bottle_price,
            AVG(COALESCE(total_amount_sold, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7_total_amount_sold,
            AVG(COALESCE(avg_bottle_price, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7_avg_bottle_price,
            AVG(COALESCE(total_amount_sold, 0)) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS ma30_total_amount_sold,
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
        ROUND(ma3_total_amount_sold, 2) AS ma3_total_amount_sold,
        ROUND(ma3_avg_bottle_price, 2) AS ma3_avg_bottle_price,
        ROUND(ma7_total_amount_sold, 2) AS ma7_total_amount_sold,
        ROUND(ma7_avg_bottle_price, 2) AS ma7_avg_bottle_price,
        ROUND(ma30_total_amount_sold, 2) AS ma30_total_amount_sold,
        ROUND(ma30_avg_bottle_price, 2) AS ma30_avg_bottle_price,
        ROUND(LAST_VALUE(temperature IGNORE NULLS) OVER (PARTITION BY item_name ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS temperature,
        (CASE WHEN rainfall IS NULL THEN 0 ELSE ROUND(rainfall, 2) END) AS rainfall,
        (CASE WHEN snowfall IS NULL THEN 0 ELSE ROUND(snowfall, 2) END) AS snowfall,
        CAST(lag_1_total_amount_sold AS INT64) AS lag_1_total_amount_sold,
        CAST(lag_7_total_amount_sold AS INT64) AS lag_7_total_amount_sold
    FROM
        final_features
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
