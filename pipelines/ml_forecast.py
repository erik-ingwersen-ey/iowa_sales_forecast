import pandas as pd
from google.cloud.bigquery import Client

from iowa_forecast.load_data import get_training_data
from iowa_forecast.ml_eval import evaluate_models, multi_evaluate_predictions
from iowa_forecast.ml_train import create_models_for_items
from iowa_forecast.plots import plot_historical_and_forecast


PROJECT_ID = "ee-ingwersen"
client = Client(project=PROJECT_ID)

HORIZON = 30
CONFIDENCE_LEVEL = 0.8
MAX_ITEMS = 6
END_DATE = pd.Timestamp.today() - pd.DateOffset(days=14)
START_DATE = END_DATE - pd.DateOffset(years=3)

TEST_START_DATE = END_DATE + pd.DateOffset(days=1)

END_DATE = END_DATE.strftime("%Y-%m-%d")
START_DATE = START_DATE.strftime("%Y-%m-%d")
TEST_START_DATE = TEST_START_DATE.strftime("%Y-%m-%d")

df_train = get_training_data(client, start_date=START_DATE, end_date=END_DATE)
items_list = df_train.groupby("item_name")["total_amount_sold"].sum().sort_values(ascending=False).index.to_list()
df_test = get_training_data(client, start_date=TEST_START_DATE, table_name="bqmlforecast.test_data", items_list=items_list)
create_models_for_items(client, items_list, max_items=MAX_ITEMS)
eval_df = evaluate_models(client, items_list[:MAX_ITEMS], horizon=HORIZON)
preds_dict = multi_evaluate_predictions(client, items_list[:6], confidence_level=CONFIDENCE_LEVEL, horizon=HORIZON)

for item_name, preds_info in preds_dict.items():
    actual_df = preds_info["train_df"].sort_values("date")
    predictions_df = preds_info["eval_df"].sort_values("date")

    plot_historical_and_forecast(
        input_timeseries=actual_df,
        timestamp_col_name="date",
        data_col_name="total_amount_sold",
        forecast_output=predictions_df,
        actual=predictions_df,
        title=f"Item: {item_name}",
        plotstartdate=(pd.to_datetime(END_DATE) - pd.DateOffset(months=2)).strftime("%Y-%m-%d"),
        prop={'size': 12},
        engine="plotly",
    )
