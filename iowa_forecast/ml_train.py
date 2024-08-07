from __future__ import annotations

import time
from typing import List

import google.cloud.bigquery
from rich.progress import track

from iowa_forecast.utils import normalize_item_name


def create_model_query(
    item_name: str,
    timestamp_col: str = "date",
    time_series_data_col: str = "total_amount_sold",
    model_name: str = "bqmlforecast.arima_plus_xreg_model",
    train_table_name: str = "bqmlforecast.training_data",
    holiday_region: str = "US",
    auto_arima: bool = True,
    adjust_step_changes: bool = True,
    clean_spikes_and_dips: bool = True,
) -> str:
    """Generates the CREATE MODEL query for the specified item."""
    item_name_norm = normalize_item_name(item_name)
    return f"""
    CREATE OR REPLACE MODEL `{model_name}_{item_name_norm}`
    OPTIONS(
      MODEL_TYPE='ARIMA_PLUS_XREG',
      TIME_SERIES_TIMESTAMP_COL='{timestamp_col}',
      TIME_SERIES_DATA_COL='{time_series_data_col}',
      HOLIDAY_REGION='{holiday_region}',
      AUTO_ARIMA={auto_arima},
      ADJUST_STEP_CHANGES={adjust_step_changes},
      CLEAN_SPIKES_AND_DIPS={clean_spikes_and_dips}
    ) AS
    SELECT
        *
    FROM
        `{train_table_name}`
    WHERE
        item_name = '{item_name}'
    ORDER BY
        date ASC
    """


def execute_query_with_retries(client, query: str, max_retries: int = 3) -> None:
    """Executes a BigQuery SQL query with retries in case of failure."""
    tries = 0
    success = False
    while not success and tries < max_retries:
        try:
            query_job = client.query(query)
            query_job.result()
            success = True
        except Exception as exc:
            tries += 1
            sleep_time = 120 * tries
            print(exc)
            print(f"Attempt {tries} failed. Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)


def create_models_for_items(
    client: google.cloud.bigquery.Client,
    items_list: List[str],
    max_items: int | None = None,
    timestamp_col: str = "date",
    time_series_data_col: str = "total_amount_sold",
    model_name: str = "bqmlforecast.arima_plus_xreg_model",
    train_table_name: str = "bqmlforecast.training_data",
    holiday_region: str = "US",
    auto_arima: bool = True,
    adjust_step_changes: bool = True,
    clean_spikes_and_dips: bool = True,
) -> None:
    """Creates models for the specified items, with progress tracking."""
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
            holiday_region,
            auto_arima,
            adjust_step_changes,
            clean_spikes_and_dips,
        )
        execute_query_with_retries(client, query)


def create_future_table(client, table_name, model_name, confidence_level=0.9, horizon=7):
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


def create_future_tables(client, columns, model: str = "bqmlforecast.arima_model", confidence_level=0.9, horizon=7):
    for column in track(columns, description="Saving ARIMA forecasts..."):
        model_name = f"{model}_{column}"
        table_name = f"bqmlforecast.forecast_{column}"
        create_future_table(client, table_name, model_name, confidence_level=confidence_level, horizon=horizon)


def train_arima_models(
    client,
    columns,
    model: str = "bqmlforecast.arima_model",
    train_table_name: str = "bqmlforecast.training_data",
    time_series_timestamp_col: str = "date",
    time_series_id_col: str = "item_name",
    confidence_level=0.9,
    horizon=7,
):
    for column in track(columns, description="Creating ARIMA models..."):
        model_name = f"{model}_{column}"
        train_arima_query = f"""
        CREATE OR REPLACE MODEL `{model_name}`
        OPTIONS
          (model_type = 'ARIMA_PLUS',
           time_series_timestamp_col = '{time_series_timestamp_col}',
           time_series_data_col = '{column}',
           time_series_id_col = '{time_series_id_col}'
          ) AS
        SELECT date, {column}, item_name
        FROM `{train_table_name}`
        """
        train_arima_job = client.query(train_arima_query)
        train_arima_job.result()

        table_name = f"bqmlforecast.forecast_{column}"
        create_future_table(client, table_name, model_name,
                            confidence_level=confidence_level, horizon=horizon)

