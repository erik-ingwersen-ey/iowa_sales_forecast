Here's a sample `installation.rst` file tailored to your Iowa Liquor Sales Forecast project:

Installation
============

This guide provides instructions on how to install and set up the Iowa Liquor Sales Forecast project.

Requirements
------------

Before you begin, ensure you have the following installed:

- [Python 3.9](https://www.python.org/downloads/release/python-390/) or higher
- [Google Cloud SDK (gcloud)](https://cloud.google.com/sdk/docs/install) installed and configured
- [Docker](https://www.docker.com/) (for containerized environments)
- [Git](https://git-scm.com/) (for cloning the repository)

Python Dependencies
--------------------

The project requires several Python packages to run.
These dependencies are listed in the `requirements.txt` file.

To install these dependencies, you can use `pip`:

```bash
pip install -r requirements.txt
```

Google Cloud Setup
-------------------

This project utilizes Google Cloud BigQuery for data storage and retrieval.
Follow these steps to set up your Google Cloud environment:

1. **Create a Google Cloud Project**:
    - Go to the [Google Cloud Console](https://console.cloud.google.com/).
    - Create a new project or select an existing one.

2. **Enable the BigQuery API**:
    - Navigate to [BigQuery API](https://console.cloud.google.com/bigquery) and enable it for your project.

3. **Set up Authentication**:
    - Navigate to [IAM Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)
      and create a service account in your project.
    - In the projects service accounts page, click on the service account
      you've created and then navigate to the "Keys" tab.
    - Select the option "ADD KEY" followed by "Create New Key".
    - Choose JSON "Key type" and click the "Create" button to download the JSON
      key file
    - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the
      path of this file:

    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"
    ```

4. **Create a BigQuery Dataset**:
    - In the BigQuery console, create a new dataset where the data and models will be stored.

Docker Setup (Optional)
-----------------------

If you prefer to run the project in a Docker container to ensure a consistent environment, follow these steps:

1. **Build the Docker Image**:

    In the root directory of the project, run:

    ```bash
    docker build -t iowa-liquor-sales-forecast .
    ```

2. **Run the Docker Container**:

    Start the container with:

    ```bash
    docker run -it --rm -e GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json" \
    -v $(pwd):/app iowa-liquor-sales-forecast
    ```

    Replace `/path/to/your/service-account-file.json` with the actual path to your credentials file.
    The `-v $(pwd):/app` option mounts the current directory inside the container.

Running the Project
--------------------

Once all dependencies are installed and your environment is set up,
you can start using the project. Here’s how:

1. **Clone the Repository**:

    If you haven't already, clone the project repository:

    ```bash
    git clone https://github.com/yourusername/iowa-liquor-sales-forecast.git
    cd iowa-liquor-sales-forecast
    ```

2. **Run the Pipeline Script**:

    Execute the `train_model_and_forecast_sales.py` script to train the
    models and start generating forecasts:

    ```bash
    python pipelines/train_model_and_forecast_sales.py
    ```

    .. attention::

        Make sure your environment variables are set correctly before running the script.

Troubleshooting
---------------

If you encounter any issues during installation or setup, here are some common solutions:

- **Missing Python Packages**:
    - Ensure that all dependencies are installed via `pip install -r requirements.txt`.

- **Google Cloud Authentication Errors**:
    - Verify that the `GOOGLE_APPLICATION_CREDENTIALS` environment variable is correctly set and points to the valid JSON key file.

- **Docker Issues**:
    - Ensure Docker is running and your system has enough resources allocated to Docker.

For further assistance, refer to the project’s [documentation](index.html).
