# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Copy the credentials file into the container
COPY ./iowa-sales-forecast-service-account.json /gcloud/application_default_credentials.json

# Update pip
RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install the iowa_forecast package to the container.
RUN pip install -e .

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variables

# Project ID where the tables and models will be saved to inside BigQuery
ENV PROJECT_ID "iowa-liquor-sales-forecast-v4"

# Dataset name where the tables and models will be stored to
ENV DATASET_NAME "bqmlforecast"

# Set the environment variable to point to the credentials file
ENV GOOGLE_APPLICATION_CREDENTIALS="/gcloud/application_default_credentials.json"


# Run app.py when the container launches
ENTRYPOINT ["python", "pipelines/train_model_and_forecast_sales.py"]
