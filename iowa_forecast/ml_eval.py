from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from google.cloud import bigquery
from rich.progress import track

from iowa_forecast.utils import normalize_item_name


def evaluate_models(
    client: bigquery.Client,
    items_list: List[str],
    end_date: str | None = None,
    horizon: int = 7,
    perform_aggregation: bool = True,
    model: str = "bqmlforecast.arima_plus_xreg_model",
    train_table_name: str = "bqmlforecast.training_data",
    actual_table_name: str = "bqmlforecast.test_data",
) -> pd.DataFrame:

    perform_aggregation = "TRUE" if perform_aggregation else "FALSE"
    eval_dfs = []

    for item_name in track(items_list, description="Evaluating models..."):
        if end_date is None:
            xdf = get_train_data(client, items_list[0], table_name=train_table_name)
            end_date = xdf["date"].max().strftime("%Y-%m-%d")

        item_name_norm = normalize_item_name(item_name)
        eval_query = f"""
        SELECT
          *
        FROM
          ML.EVALUATE(
            MODEL `{model}_{item_name_norm}`,
            (
              SELECT * FROM `{actual_table_name}` WHERE item_name = "{item_name}" AND date > DATE('{end_date}')
            ),
            STRUCT(
              {perform_aggregation} AS perform_aggregation,
              {horizon} AS horizon))
        """
        eval_df = get_data(client, eval_query).assign(**{"item_name": item_name})
        eval_dfs.append(eval_df)

    eval_df = pd.concat(eval_dfs)
    eval_df = eval_df[eval_df.columns[::-1]]
    return eval_df


def get_data(client: bigquery.Client, query: str) -> pd.DataFrame:
    """Executes a BigQuery SQL query and returns the result as a DataFrame."""
    query_job = client.query(query)
    return query_job.to_dataframe()


def create_query(
    table: str, item_name: str, date_filter: str = None, order_by: str = None
) -> str:
    """Creates a SQL query string."""
    query = f"SELECT * FROM `{table}` WHERE item_name = '{item_name}'"
    if date_filter:
        query += f" AND {date_filter}"
    if order_by:
        query += f" ORDER BY {order_by}"
    return query


def get_train_data(
    client: bigquery.Client,
    item_name: str,
    table_name: str = "bqmlforecast.training_data",
    order_by: str = "item_name, date",
    date_filter: str = None,
) -> pd.DataFrame:
    """Retrieves training data for the specified item."""
    query = create_query(
        table=table_name,
        item_name=item_name,
        order_by=order_by,
        date_filter=date_filter,
    )
    return get_data(client, query)


def get_actual_data(
    client: bigquery.Client,
    item_name: str,
    end_date: str,
    horizon: int = 7,
    table_name: str = "bqmlforecast.test_data",
    order_by: str = "item_name, date",
) -> pd.DataFrame:
    """Retrieves actual data for the specified item and date range."""
    # date_filter = f"date BETWEEN DATE_ADD('{end_date}', INTERVAL 1 DAY) AND DATE_ADD('{end_date}', INTERVAL {1 + horizon} DAY)"
    # query = create_query(
    #     table=table_name,
    #     item_name=item_name,
    #     date_filter=date_filter,
    #     order_by=order_by,
    # )
    query = f'SELECT * FROM `{table_name}` WHERE item_name = "{item_name}" ORDER BY {order_by}'
    return get_data(client, query).astype({"date": str})


def get_predictions(
    client: bigquery.Client,
    item_name: str,
    end_date: str,
    model: str = "bqmlforecast.arima_plus_xreg_model",
    forecast_table_name: str = "bqmlforecast.forecast_data",
    horizon: int = 7,
    confidence_level: float = 0.8,
) -> pd.DataFrame:
    """Retrieves forecast predictions for the specified item."""
    item_name_norm = normalize_item_name(item_name)
    predictions_query = f"""
    SELECT *
    FROM ML.FORECAST(
        MODEL `{model}_{item_name_norm}`,
        STRUCT({horizon} AS horizon, {confidence_level} AS confidence_level),
        (
          SELECT * FROM {forecast_table_name} WHERE item_name = "{item_name}" AND date >= DATE('{end_date}') ORDER BY date ASC
        )
    )
    """
    predictions_df = get_data(client, predictions_query)
    predictions_df = predictions_df.rename(columns={"forecast_timestamp": "date"})
    predictions_df["date"] = pd.to_datetime(predictions_df["date"]).dt.strftime(
        "%Y-%m-%d"
    )
    predictions_df["item_name"] = item_name
    return predictions_df


def evaluate_predictions(
    client: bigquery.Client,
    item_name: str,
    end_date: str | None = None,
    model: str = "arima_plus_xreg_model",
    actual_table_name: str = "bqmlforecast.test_data",
    train_table_name: str = "bqmlforecast.training_data",
    forecast_table_name: str = "bqmlforecast.forecast_data",
    horizon: int = 7,
    confidence_level: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluates the predictions against actual data."""

    train_df = get_train_data(client, item_name, table_name=train_table_name)

    if end_date is None:
        end_date = train_df["date"].max().strftime("%Y-%m-%d")

    if hasattr(end_date, "strftime"):
        end_date = end_date.strftime("%Y-%m-%d")

    actual_df = get_actual_data(
        client, item_name, end_date, horizon, table_name=actual_table_name
    )
    predictions_df = get_predictions(
        client,
        item_name,
        end_date,
        forecast_table_name=forecast_table_name,
        horizon=horizon,
        confidence_level=confidence_level,
    )
    forecast_df = actual_df.merge(
        predictions_df[
            [
                "date",
                "item_name",
                "forecast_value",
                "prediction_interval_lower_bound",
                "prediction_interval_upper_bound",
                "confidence_level",
            ]
        ],
        on=["date", "item_name"],
        how="outer",
    )
    forecast_df["forecast_value"] = np.where(forecast_df["forecast_value"] < 0, 0, forecast_df["forecast_value"])
    forecast_df["prediction_interval_lower_bound"] = np.where(
        forecast_df["prediction_interval_lower_bound"] < 0, 0, forecast_df["prediction_interval_lower_bound"]
    )
    return train_df, forecast_df


def multi_evaluate_predictions(
    client: bigquery.Client,
    items_list: List[str],
    end_date: str | None = None,
    model: str = "arima_plus_xreg_model",
    actual_table_name: str = "bqmlforecast.test_data",
    train_table_name: str = "bqmlforecast.training_data",
    forecast_table_name: str = "bqmlforecast.forecast_data",
    horizon: int = 7,
    confidence_level: float = 0.8,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    results_dict = {}
    for item_name in track(items_list, description="Generating predictions..."):
        train_df, eval_df = evaluate_predictions(
            client,
            item_name,
            end_date,
            model,
            actual_table_name,
            train_table_name,
            forecast_table_name,
            horizon,
            confidence_level,

        )
        results_dict[item_name] = {
            "train_df": train_df,
            "eval_df": eval_df,
        }

    return results_dict


def explain_model(
    client: bigquery.Client,
    item_name: str,
    table_name: str = "bqmlforecast.training_data",
    model: str = "bqmlforecast.arima_plus_xreg_model",
    horizon: int = 7,
    confidence_level: float = 0.8,
    order_by: str | None = "date",
    date_filter: str | None = None,
):
    item_name_norm = normalize_item_name(item_name)
    data_query = create_query(
        table=table_name,
        item_name=item_name,
        order_by=order_by,
        date_filter=date_filter,
    )

    query = f"""
    SELECT
        *
    FROM
        ML.EXPLAIN_FORECAST(
            MODEL `{model}_{item_name_norm}`,
            STRUCT({horizon} AS horizon, {confidence_level} AS confidence_level),
            ({data_query})
        )
    """
    return get_data(client, query).assign(**{"item_name": item_name})


def explain_models(
    client: bigquery.Client,
    items_list: List[str],
    table_name: str = "bqmlforecast.training_data",
    model: str = "bqmlforecast.arima_plus_xreg_model",
    horizon: int = 7,
    confidence_level: float = 0.8,
    order_by: str | None = "date",
    date_filter: str | None = None,
) -> Dict[str, pd.DataFrame]:
    explain_dfs = {}
    for item_name in track(items_list, description="Generating explanations..."):
        explain_df = explain_model(
            client,
            item_name,
            table_name,
            model,
            horizon,
            confidence_level,
            order_by,
            date_filter,
        )
        explain_dfs["item_name"] = explain_df

    return explain_dfs


def get_future_sales_query(
    client,
    item_name,
    horizon: int = 7,
    train_table_name="bqmlforecast.training_data",
    end_date: str | None = None,
    state: str = "IA",
):
    if end_date is None:
        xdf = get_train_data(client, item_name, table_name=train_table_name)
        end_date = xdf["date"].max()

    end_date = pd.Timestamp(end_date)
    previous_year_date = end_date - pd.DateOffset(years=1)
    previous_year = int(previous_year_date.year)
    end_date = end_date.strftime("%Y-%m-%d")

    query = f"""
    -- Step 1: Get the maximum date from training data
    WITH max_date AS (
        SELECT MAX(date) AS max_date
        FROM `{train_table_name}`
    ),
    recent_sales_data AS (
        SELECT
            date,
            item_name,
            total_amount_sold,
            avg_bottle_price,
            total_volume_sold_liters,
            total_sale_dollars,
            avg_bottle_cost,
            lag_1_total_amount_sold,
            lag_7_total_amount_sold,
            ma3_total_amount_sold,
            ma3_avg_bottle_price,
            ma3_total_sale_dollars,
            ma7_total_amount_sold,
            ma7_avg_bottle_price,
            ma7_total_sale_dollars,
            ma30_total_amount_sold,
            ma30_avg_bottle_price,
            ma30_total_sale_dollars
        FROM
            `{train_table_name}`
        WHERE
            date <= DATE('{end_date}')
            AND item_name = "{item_name}"
    ),
    -- Step 2: Generate future dates from the maximum date to <HORIZON> days ahead
    future_dates AS (
        SELECT
            DATE_ADD(max_date.max_date, INTERVAL day_offset DAY) AS date
        FROM
            max_date,
            UNNEST(GENERATE_ARRAY(1, {horizon})) AS day_offset
    ),
    -- Step 3: Use weather from previous year
    future_weather_data AS (
        SELECT
            fd.date,
            AVG(hd.temp) AS temperature,
            AVG(CASE WHEN hd.prcp = 999.9 THEN 0 ELSE hd.prcp END) AS rainfall,
            AVG(CASE WHEN hd.sndp = 999.9 THEN 0 ELSE hd.sndp END) AS snowfall
        FROM
            `future_dates` fd
        LEFT JOIN
            `bigquery-public-data.noaa_gsod.gsod2023` hd
        ON
            DATE_ADD(hd.date, INTERVAL 1 YEAR) = fd.date
        WHERE
            stn IN (SELECT usaf FROM `bigquery-public-data.noaa_gsod.stations` WHERE state = '{state}')
        GROUP BY
            fd.date
    ),
    -- Step 4: Get the recent sales data with the maximum date
    recent_sales_data_with_max_date AS (
        SELECT
            rsd.*
        FROM
            `recent_sales_data` rsd
        JOIN
            max_date
        ON
            rsd.date = max_date.max_date
    ),
    -- Step 5: Construct the future data with the necessary features
    future_data AS (
        SELECT
            fd.date,
            rsd.item_name,
            0 AS total_amount_sold,
            rsd.avg_bottle_price,
            rsd.total_volume_sold_liters,
            rsd.total_sale_dollars,
            rsd.avg_bottle_cost,
            EXTRACT(DAYOFWEEK FROM fd.date) AS day_of_week,
            EXTRACT(WEEK FROM fd.date) AS week_of_year,
            EXTRACT(MONTH FROM fd.date) AS month,
            EXTRACT(YEAR FROM fd.date) AS year,
            rsd.ma3_total_amount_sold,
            rsd.ma3_avg_bottle_price,
            rsd.ma3_total_sale_dollars,
            rsd.ma7_total_amount_sold,
            rsd.ma7_avg_bottle_price,
            rsd.ma7_total_sale_dollars,
            rsd.ma30_total_amount_sold,
            rsd.ma30_avg_bottle_price,
            rsd.ma30_total_sale_dollars,
            COALESCE(fwd.temperature, 0) AS temperature,
            COALESCE(fwd.rainfall, 0) AS rainfall,
            COALESCE(fwd.snowfall, 0) AS snowfall,
            CAST(rsd.lag_1_total_amount_sold AS INT64) AS lag_1_total_amount_sold,
            CAST(rsd.lag_7_total_amount_sold AS INT64) AS lag_7_total_amount_sold
        FROM
            future_dates fd
        CROSS JOIN
            (SELECT DISTINCT item_name FROM `recent_sales_data`) rsd_item_names
        LEFT JOIN
            recent_sales_data_with_max_date rsd
        ON
            rsd.item_name = rsd_item_names.item_name
        LEFT JOIN
            `future_weather_data` fwd
        ON
            fd.date = fwd.date
    )
    -- Step 5: Create the future_data table
    SELECT
        *
    FROM
        future_data
    WHERE
        item_name = "{item_name}"
    ORDER BY
        date ASC
    """
    return query
