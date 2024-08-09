import os
from typing import Tuple

import pandas as pd
from google.cloud import bigquery

from iowa_forecast import load_data, ml_eval
from iowa_forecast import ml_train
from iowa_forecast.utils import (create_bigquery_table_from_pandas,
                                 create_dataset_if_not_found)


PROJECT_ID = os.environ.get("PROJECT_ID", "iowa-liquor-sales-forecast-v2")
DATASET_NAME = os.environ.get("DATASET_NAME", "bqmlforecast")
CREDENTIALS_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "./iowa-sales-forecast-service-account.json")
HORIZON = 30  # Days

# Validate the presence of CREDENTIALS_FILE
if not CREDENTIALS_FILE:
    raise EnvironmentError(
        "The GOOGLE_APPLICATION_CREDENTIALS environment variable is not set. "
        "Please ensure that you have set this variable to the path of your Google Cloud credentials JSON file. "
        "For more information, see: https://cloud.google.com/docs/authentication/getting-started\n"
        "Additional context: you have to create a service account in your Google Cloud Account, then\n"
        "export the service account credentials as a JSON file. Finally, specify the location of the downloaded\n"
        "JSON credentials on the 'COPY' command inside the Dockerfile:\n\n\t"
        "COPY <path-to-your-credentials> /gcloud/application_default_credentials.json\n\n"
        "After making this change, build a new image of the application running the command:\n\n\t"
        "docker build -t iowa-sales-forecast .\n\n"
        "Then run the generated container using the command:\n\n\t"
        "docker run -p 8080:8080 iowa-sales-forecast"
    )
elif not os.path.isfile(CREDENTIALS_FILE):
    raise FileNotFoundError(
        f"The file specified in the GOOGLE_APPLICATION_CREDENTIALS environment variable does not exist: {CREDENTIALS_FILE}. "
        "Please ensure that the file path is correct and that the file is accessible inside the Docker container. "
        "For more details on handling credentials in Docker, see: https://cloud.google.com/docs/authentication/production#passing_code"
    )


def train_models_and_generate_forecast() -> Tuple[str, int]:
    """
    Execute the data load process for training and future data.

    This function connects to BigQuery, loads the training data from a specified
    table, and creates future data for forecasting. It logs the status of the
    process and handles any exceptions that occur.

    Returns
    -------
    Tuple[str, int]
        A tuple containing a status message and an HTTP status code.
        If the process is successful, returns a success message and code 0.
        If an error occurs, returns an error message and code 500.

    Examples
    --------
    >>> message, status = train_models_and_generate_forecast()
    >>> print(message)
    Data processing, model train and testing successfully completed.
    >>> print(status)
    0
    """
    client = bigquery.Client(
        project=PROJECT_ID,
        client_options={"credentials_file": CREDENTIALS_FILE}
    )
    create_dataset_if_not_found(client, dataset_name=DATASET_NAME)

    END_DATE = pd.Timestamp.today() - pd.DateOffset(days=14)
    START_DATE = END_DATE - pd.DateOffset(years=3)
    TEST_START_DATE = END_DATE + pd.DateOffset(days=1)

    END_DATE = END_DATE.strftime("%Y-%m-%d")
    START_DATE = START_DATE.strftime("%Y-%m-%d")
    TEST_START_DATE = TEST_START_DATE.strftime("%Y-%m-%d")

    try:
        training_data = load_data.get_training_data(
            client=client,
            table_name=f"{PROJECT_ID}.{DATASET_NAME}.training_data",
            start_date=START_DATE,
            end_date=END_DATE,
        )
        # Retrieve a list of item names that exist in the training data.
        # The list contains the item names sorted by their total amount sold
        # in the whole timespan of the training data.
        # This list is used to filter the testing data, and it also limits the number
        # of multivariate ARIMA models that will be created. This imposed limit
        # was created due to limitations in the volume of bytes that can be
        # processed without specifying a billing account on Google Cloud.
        items_list = (
            training_data.groupby("item_name")["total_amount_sold"].sum()
            .sort_values(ascending=False).index
            .to_list()
        )
        # The testing data contains values that will be used to validade the model.
        df_test = load_data.get_training_data(
            client, start_date=TEST_START_DATE,
            table_name=f"{PROJECT_ID}.{DATASET_NAME}.test_data",
            items_list=items_list,
        )

        # To generate the forecast of future sales, we need first to predict the
        # future values for our model's features. The sales forecast model contains
        # multiple features that are, however, derived from the columns defined
        # in the list below. Therefore, we'll be training a simple ARIMA model for
        # each of these features and then forecasting their future values
        # and use these values along with the test and training data to generate
        # the rest of the necessary features. Additional features derived from
        # these three columns include moving averages and lag columns
        columns = ["avg_bottle_price", "avg_bottle_cost", "total_volume_sold_liters"]
        ml_train.train_arima_models(
            client,
            columns,
            model=f"{PROJECT_ID}.{DATASET_NAME}.arima_model",
            train_table_name=f"{PROJECT_ID}.{DATASET_NAME}.training_data",
            test_table_name=f"{PROJECT_ID}.{DATASET_NAME}.test_data",
            model_metrics_table_name=None,
            time_series_timestamp_col="date",
            time_series_id_col="item_name",
            confidence_level=0.9,
            horizon=HORIZON,
        )

        load_data.create_future_feature_tables(
            client,
            columns=columns,
            model=f"{PROJECT_ID}.{DATASET_NAME}.arima_model",
            table_base_name=f"{PROJECT_ID}.{DATASET_NAME}.forecast",
            confidence_level=0.9,
            horizon=HORIZON,
        )

        load_data.create_future_data(
            client=client,
            train_table_name=f"{PROJECT_ID}.{DATASET_NAME}.training_data",
            test_table_name=f"{PROJECT_ID}.{DATASET_NAME}.test_data",
            forecast_table_name=f"{PROJECT_ID}.{DATASET_NAME}.future_forecast_data",
            horizon=HORIZON,
            dataset_id=DATASET_NAME,
        )
        ml_train.create_models_for_items(client, items_list, max_items=4, clean_spikes_and_dips=True)
        forecast_dict = ml_eval.multi_evaluate_predictions(
            client,
            items_list[:4],
            confidence_level=0.9,
            horizon=HORIZON,
            forecast_table_name=f"{PROJECT_ID}.{DATASET_NAME}.future_forecast_data",
        )
        items_predictions = []
        for item_name, forecast_info in forecast_dict.items():
            training_df = forecast_info["train_df"].sort_values("date").assign(
                **{
                    "forecast_value": 0,
                    "prediction_interval_lower_bound": 0,
                    "prediction_interval_upper_bound": 0,
                    "confidence_level": 0,
                    "time_series_type": "history",
                }
            )
            predictions_df = forecast_info["eval_df"].sort_values("date").assign(
                **{"time_series_type": "forecast"}
            )
            items_predictions = [*items_predictions, training_df, predictions_df]

        items_predictions_df = (
            pd.concat(items_predictions, ignore_index=True)
            .sort_values(["item_name", "date"])
        )
        items_predictions_df = items_predictions_df.fillna(0).astype(
            {
                col: str for col in
                items_predictions_df.select_dtypes(include=object).columns
            }
        )
        create_bigquery_table_from_pandas(
            client,
            items_predictions_df,
            "predictions_table",
            DATASET_NAME,
        )

        return "Data processing, model train and testing successfully completed.", 0

    except Exception as exc:
        print(f"Process failed: {exc}")
        return f"Process failed: {str(exc)}", 500


if __name__ == "__main__":
    train_models_and_generate_forecast()
