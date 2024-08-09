"""
BigQuery Model Evaluation and Forecasting Module.

This module provides a set of functions for evaluating and forecasting
time series data using models in Google BigQuery. It includes utilities
for querying, evaluating, and explaining models, as well as for
aggregating results across multiple items.

Functions
---------
* `evaluate_models`: evaluates 'ARIMA_PLUS_XREG' models for a list of items,
  returning performance metrics in a `pandas.DataFrame`.

* `get_data`: execute a BigQuery SQL query and return the result as a `pandas.DataFrame`.

* `create_query`: creates an SQL query string based on the specified parameters.

* `get_train_data`: retrieve training data for a specified item from BigQuery.

* `get_actual_data`: retrieve actual test data for a specified item and date range
  from BigQuery.

* `get_predictions`: retrieve forecast predictions for a specified item using a
  BigQuery model.

* `evaluate_predictions`: evaluate forecast predictions against actual data
  and return comparison `pandas.DataFrames`.

* `multi_evaluate_predictions`: evaluate predictions for multiple items and return
  a dictionary of results.

* `explain_model`: generate explanations for forecast predictions using a BigQuery
  model and return the result as a `pandas.DataFrame`.

Notes
-----
This module is designed to work with Google BigQuery and requires a valid
BigQuery client instance. The models evaluated and forecasted by this
module are primarily intended for time series forecasting in various
business contexts.

See Also
--------
Google BigQuery: https://cloud.google.com/bigquery
BigQuery ML: https://cloud.google.com/bigquery-ml
"""
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
    """
    Evaluate 'ARIMA_PLUS_XREG' models for a list of items.

    This function evaluates models for a list of items using BigQuery
    and returns a DataFrame containing the evaluation results.

    Parameters
    ----------
    client : bigquery.Client
        An instance of the BigQuery client used to execute queries.
    items_list : List[str]
        A list of item names for which models should be evaluated.
    end_date : str or None, optional
        The end date for the evaluation period.
        If None, the maximum date from the training data is used.
    horizon : int, default=7
        The number of time steps (days) ahead to evaluate.
    perform_aggregation : bool, default=True
        Whether to perform aggregation in the evaluation.
    model : str, default="bqmlforecast.arima_plus_xreg_model"
        The base name of the model.
    train_table_name : str, default="bqmlforecast.training_data"
        The name of the table containing training data.
    actual_table_name : str, default="bqmlforecast.test_data"
        The name of the table containing actual data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the evaluation metrics for each item.
    """
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
    eval_df = eval_df
    return eval_df


def get_data(client: bigquery.Client, query: str) -> pd.DataFrame:
    """
    Execute a BigQuery SQL query and return the result as a DataFrame.

    Parameters
    ----------
    client : bigquery.Client
        An instance of the BigQuery client used to execute the query.
    query : str
        The SQL query to be executed.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the query result.
    """
    query_job = client.query(query)
    return query_job.to_dataframe()


def create_query(
    table: str,
    item_name: str,
    date_filter: str | None = None,
    order_by: str | None = None,
) -> str:
    """Create an SQL query string for the specified table and item.

    Parameters
    ----------
    table : str
        The name of the table from which to select data.
    item_name : str
        The item name to filter the data by.
    date_filter : str, optional
        A query component to filter the dates retrieved from the database.
    order_by : str, optional
        A column or list of column names to order the results by.
        For example, if you want to sort by 'date' and 'item_name',
        you can specify the following value: `'date, item_name'`.

    Returns
    -------
    str
        An SQL query string.
    """
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
    date_filter: str | None = None,
) -> pd.DataFrame:
    """
    Retrieve training data for the specified item from BigQuery.

    Parameters
    ----------
    client : bigquery.Client
        An instance of the BigQuery client used to execute the query.
    item_name : str
        The name of the item for which to retrieve training data.
    table_name : str, default="bqmlforecast.training_data"
        The name of the table containing the training data.
    order_by : str, default="item_name, date"
        The column(s) to order the results by.
    date_filter : str, optional
        A filter for the date column.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the training data for the specified item.
    """
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
    """
    Retrieve actual data for the specified item and date range.

    Parameters
    ----------
    client : bigquery.Client
        An instance of the BigQuery client used to execute the query.
    item_name : str
        The name of the item for which to retrieve actual data.
    end_date : str
        The end date for the actual data retrieval.
    horizon : int, default=7
        The number of time steps (days) ahead to retrieve.
    table_name : str, default="bqmlforecast.test_data"
        The name of the table containing the actual data.
    order_by : str, default="item_name, date"
        The column(s) to order the results by.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the actual data for the specified item.
    """
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
    """Retrieve forecast predictions for the specified item.

    Parameters
    ----------
    client : bigquery.Client
        An instance of the BigQuery client used to execute the query.
    item_name : str
        The name of the item for which to retrieve forecast predictions.
    end_date : str
        The end date for the forecast period.
    model : str, default="bqmlforecast.arima_plus_xreg_model"
        The base name of the model.
    forecast_table_name : str, default="bqmlforecast.forecast_data"
        The name of the table containing the forecast data.
    horizon : int, default=7
        The number of time steps (days) ahead to forecast.
    confidence_level : float, default=0.8
        The confidence level for the forecast predictions.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the forecast predictions.
    """
    item_name_norm = normalize_item_name(item_name)
    predictions_query = f"""
    SELECT *
    FROM ML.FORECAST(
        MODEL `{model}_{item_name_norm}`,
        STRUCT({horizon} AS horizon, {confidence_level} AS confidence_level),
        (
          SELECT * FROM {forecast_table_name} WHERE item_name = "{item_name}" AND date >= DATE('{end_date}') ORDER BY date
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
    """
    Evaluate forecast predictions against actual data.

    This function compares forecast predictions from the model with
    actual data, returning DataFrames for the training data and the
    evaluated forecast.

    Parameters
    ----------
    client : bigquery.Client
        An instance of the BigQuery client used to execute the query.
    item_name : str
        The name of the item to evaluate.
    end_date : str or None, optional
        The end date for the evaluation period. If None, the maximum date
        from the training data is used.
    model : str, default="arima_plus_xreg_model"
        The base name of the model.
    actual_table_name : str, default="bqmlforecast.test_data"
        The name of the table containing actual data.
    train_table_name : str, default="bqmlforecast.training_data"
        The name of the table containing training data.
    forecast_table_name : str, default="bqmlforecast.forecast_data"
        The name of the table containing forecast data.
    horizon : int, default=7
        The number of time steps ahead to evaluate.
    confidence_level : float, default=0.8
        The confidence level for the forecast evaluation.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two DataFrames: one with the training data
        and one with the evaluated forecast data.
    """
    train_df = (
        get_train_data(client, item_name, table_name=train_table_name)
        .assign(**{"time_series_type": "history"})
    )

    if end_date is None:
        end_date = train_df["date"].max().strftime("%Y-%m-%d")

    if hasattr(end_date, "strftime"):
        end_date = end_date.strftime("%Y-%m-%d")

    actual_df = get_actual_data(
        client, item_name, end_date, horizon, table_name=actual_table_name
    ).assign(**{"time_series_type": "test"})

    predictions_df = get_predictions(
        client,
        item_name,
        end_date,
        forecast_table_name=forecast_table_name,
        horizon=horizon,
        confidence_level=confidence_level,
    ).assign(**{"time_series_type": "forecast"})

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
    forecast_df["forecast_value"] = np.where(
        forecast_df["forecast_value"] < 0,
        0,
        forecast_df["forecast_value"],
    )
    forecast_df["prediction_interval_lower_bound"] = np.where(
        forecast_df["prediction_interval_lower_bound"] < 0,
        0,
        forecast_df["prediction_interval_lower_bound"],
    )
    forecast_df["prediction_interval_upper_bound"] = np.where(
        forecast_df["prediction_interval_upper_bound"] < 0,
        0,
        forecast_df["prediction_interval_upper_bound"],
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
    """
    Evaluate predictions for multiple items and return results as a dictionary.

    This function evaluates forecast predictions against actual data for
    multiple items, returning a dictionary of results for each item.

    Parameters
    ----------
    client : bigquery.Client
        An instance of the BigQuery client used to execute the query.
    items_list : List[str]
        A list of item names to evaluate.
    end_date : str or None, optional
        The end date for the evaluation period. If None, the maximum date
        from the training data is used.
    model : str, default="arima_plus_xreg_model"
        The base name of the model.
    actual_table_name : str, default="bqmlforecast.test_data"
        The name of the table containing actual data.
    train_table_name : str, default="bqmlforecast.training_data"
        The name of the table containing training data.
    forecast_table_name : str, default="bqmlforecast.forecast_data"
        The name of the table containing forecast data.
    horizon : int, default=7
        The number of time steps (days) ahead to evaluate.
    confidence_level : float, default=0.8
        The confidence level for the forecast evaluation.

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        A dictionary where each key is an item name and the value is a
        dictionary containing two DataFrames: one for the training data
        and one for the evaluated forecast data.

        * Keys: names of each item in the dictionary.
        * Sub-keys: `'train_df'`, `'eval_df'`

    """
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
    """
    Generate explanations for forecast predictions using a BigQuery model.

    This function explains the forecast predictions generated by a model,
    returning the results as a `pandas.DataFrame`.

    Parameters
    ----------
    client : bigquery.Client
        An instance of the BigQuery client used to execute the query.
    item_name : str
        The name of the item for which to generate explanations.
    table_name : str, default="bqmlforecast.training_data"
        The name of the table containing the data used to generate
        explanations.
    model : str, default="bqmlforecast.arima_plus_xreg_model"
        The base name of the model.
    horizon : int, default=7
        The number of time steps (days) ahead to explain.
    confidence_level : float, default=0.8
        The confidence level for the explanations.
    order_by : str | None, default='date'
        A column name to order the results by.
    date_filter : str | None, optional
        A filter for the 'date' column.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the explanations for the forecast predictions.

    See Also
    --------
    ML.EXPLAIN_FORECAST: https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-explain-forecast
    """
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
