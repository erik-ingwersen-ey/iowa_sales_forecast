# Iowa Liquor Sales Forecast

This repository contains the functions created to generate a sales forecasting
model that predicts sales based on the historical data of liquor purchases from
the state of Iowa.

The created model consists of a multivariate ARIMA model that includes
relevant features such as moving averages of key columns from the dataset, 
lag columns and weather forecast information.

All data used to train the model was obtained from the library
of [BigQuery public datasets](https://cloud.google.com/bigquery/public-data).

All the datasets and models created are stored inside **BigQuery**.
Therefore, to run this solution and generate the sales forecasts,
you need to [register an account in Google Cloud](https://console.cloud.google.com/).
Then you have to [create a new project](https://developers.google.com/workspace/guides/create-project),
[enable the BigQuery service](https://cloud.google.com/bigquery/docs/enable-transfer-service)
to your account and configure your credentials.

## Forecast Results

A report with the latest forecast results can be found at:
[Iowa Liquor Sales Forecast Report](https://lookerstudio.google.com/reporting/df348e6b-5d25-47bd-ae51-d7d40906a73b)

## Documentation

Please refer to the [Iowa Sales Forecast Documentation](https://erik-ingwersen-ey.github.io/iowa_sales_forecast)
page for more information about the project.

## Code Walkthrough

You can find a step-by-step walkthrough of the entire solution, including 
the data extraction, feature engineering, and transformation, model training
and evaluation, as well as forecasting future sales at:
[notebooks/Walkthrough.ipynb](./notebooks/Walkthrough.ipynb)

## Pipelines

The [pipelines](./pipelines) folder contains scripts that can be used as
entrypoints to perform several tasks related to the solution.

## Additional Information

### Docker Container

The [Dockerfile](./Dockerfile) defines the Docker container configuration to
replicate the environment used to develop and run the forecasting model.
By using this Docker container, you can ensure that the code runs consistently
across different environments. 

To build and run the Docker container, you can use the following commands:

* **Build the Docker image:**
    
  ```bash
  docker build -t iowa-liquor-sales-forecast .
  ```

* **Run the Docker container:**
    
  ```bash
  docker run -it --rm iowa-liquor-sales-forecast
  ```

### Environment Variables

The solution relies on a few environment variables that need to be set up for proper operation.
These include:

- `GOOGLE_APPLICATION_CREDENTIALS`: Path to the JSON file that contains your Google Cloud service account credentials.
- `PROJECT_ID`: The ID of your Google Cloud project.
- `DATASET_ID`: The ID of the BigQuery dataset where the data is stored.

You can set these environment variables in your shell or define them in a `.env` file,
which will be automatically loaded when running the Docker container or scripts.

### Testing

The repository includes a suite of tests to ensure that the code behaves as expected.
You can run the tests using `pytest`:

```bash
# Run tests
pytest tests/
```

### Continuous Integration (CI)

This repository is set up with a Continuous Integration (CI) pipeline using GitHub Actions.
The CI pipeline is configured to run the tests automatically whenever code is pushed to the
repository or a pull request is created. This helps to ensure that new changes do not break existing
functionality. It also contains a pipeline that recreates the documentation 
for the project and generates a new release of the documentation on GitHub 
Pages.

Here's the list of currently available pipelines for the project:

* [deploy-docs.yml](./.github/workflows/deploy-docs.yml): deploy
  documentation to GitHub Pages.
* [test-code.yml](./.github/workflows/test-code.yml): run the unit-tests 
  from the [tests](./tests) directory and generate a test coverage report
  for the project.


### License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.


### Codebase Static Test Results

The `iowa_forecast` package received the following pylint scores:

* `iowa_forecast/__init__.py`: 10.0
* `iowa_forecast/models_configs.py`: 10.0
* `iowa_forecast/ml_train.py`: 10.0
* `iowa_forecast/plots.py`: 9.8
* `iowa_forecast/utils.py`: 8.99
* `iowa_forecast/load_data.py`: 9.31
* `iowa_forecast/ml_eval.py`: 8.41

* **Average Score:** 9.50