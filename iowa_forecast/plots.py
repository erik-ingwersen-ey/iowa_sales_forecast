"""
Time Series Plotting and Date Handling Module.

This module provides functions for handling date-related operations
on DataFrames and for visualizing time series data, including historical,
forecast, and actual values. It supports both Matplotlib and Plotly
as plotting engines, offering flexibility in visualization options.

Functions
---------
* `convert_to_datetime`: convert a column in a DataFrame to datetime format.
* `filter_by_date`: filter a DataFrame by a start date.
* `plot_historical_and_forecast`: plot historical data with optional forecast and actual values.

Notes
-----
This module is designed to assist in the preparation and visualization
of time series data. The `plot_historical_and_forecast` function is
particularly useful for comparing historical data with forecasted
and actual values, with options to highlight peaks and add custom
plot elements using either Matplotlib or Plotly.

"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import find_peaks


def convert_to_datetime(dataframe: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Convert a specified column in a DataFrame to datetime format.

    This function takes a DataFrame and converts the specified column
    to pandas' datetime format, enabling datetime operations on that column.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the column to convert.
    col : str
        The name of the column in the DataFrame to convert to datetime format.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with the specified column converted to datetime format.

    Notes
    -----
    You can also chain this function using `pandas.DataFrame.pipe`:

    .. code-block:: python

        df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'value': [10, 15]
        }).pipe(convert_to_datetime, 'date')

    Examples
    --------
    Convert the 'date' column in a DataFrame to datetime format:

    >>> df = pd.DataFrame({
    ...     'date': ['2023-01-01', '2023-01-02'],
    ...     'value': [10, 15]
    ... })
    >>> df = convert_to_datetime(df,'date')
    >>> dataframe['date'].dtype
    dtype('<M8[ns]')
    """
    dataframe[col] = pd.to_datetime(dataframe[col])
    return dataframe


def filter_by_date(dataframe: pd.DataFrame, col: str, start_date: str):
    """Filter a DataFrame by a start date.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame to filter.
    col : str
        The name of the datetime column to filter by.
    start_date : str
        The start date to filter the DataFrame. If None, no filtering is done.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame, or the original if no filtering is applied.
    """
    datetime_start_date = pd.to_datetime(start_date, errors="coerce")
    if pd.notna(datetime_start_date):
        return dataframe[pd.to_datetime(dataframe[col]) >= datetime_start_date]
    return dataframe


def plot_series(x_data, y_data, label: str, linestyle: str = "-", **kwargs) -> None:
    """
    Plot a series of data with optional markers.

    This function plots a series of data using Matplotlib, with options
    to customize the line style, add markers, and change the marker color.

    Parameters
    ----------
    x_data : array-like
        The data for the x-axis.
    y_data : array-like
        The data for the y-axis.
    label : str
        The label for the plot legend.
    linestyle : str, default="-"
        The line style for the plot, e.g., '-' for a solid line, '--' for a dashed line.
    **kwargs : dict, optional
        Additional keyword arguments for customizing the plot.
        Available options:
        - marker: str
            The marker style for scatter points.
        - color: str
            The color of the markers.

    Returns
    -------
    None

    Examples
    --------
    Plot a series of data with default settings:

    >>> x = [1, 2, 3, 4]
    >>> y = [10, 15, 10, 20]
    >>> plot_series(x, y, label="Sample Data")

    Plot a series with markers:

    >>> plot_series(x, y, label="Sample Data", marker="o", color="red")
    """
    marker = kwargs.get("marker")
    color = kwargs.get("color")

    plt.plot(x_data, y_data, label=label, linestyle=linestyle)
    if marker:
        plt.scatter(x_data, y_data, marker=marker, color=color)


def plot_historical_and_forecast(  # pylint: disable=too-many-arguments, too-many-locals
    input_timeseries: pd.DataFrame,
    timestamp_col_name: str,
    data_col_name: str,
    forecast_output: pd.DataFrame | None = None,
    forecast_col_names: dict | None = None,
    actual: pd.DataFrame | None = None,
    actual_col_names: dict | None = None,
    title: str | None = None,
    plot_start_date: str | None = None,
    show_peaks: bool = True,
    engine: str = "matplotlib",
    **plot_kwargs,
) -> None:
    """
    Plot historical data with optional forecast and actual values.

    This function visualizes time series data with options for forecasting,
    actual values, and peak highlighting. It supports both Matplotlib
    and Plotly as plotting engines.

    Parameters
    ----------
    input_timeseries : pd.DataFrame
        The DataFrame containing historical time series data.
    timestamp_col_name : str
        The name of the column containing timestamps.
    data_col_name : str
        The name of the column containing the data values.
    forecast_output : pd.DataFrame, optional
        The DataFrame containing forecast data. Specify this parameter
        if you want to plot the historical or actual data and the forecasted
        values with lines of different colors.
    forecast_col_names : dict, optional
        Dictionary mapping forecast DataFrame columns, by default None.
        Keys: 'timestamp', 'value', 'confidence_level', 'lower_bound',
        'upper_bound'.
    actual : pd.DataFrame, optional
        The `pandas.DataFrame` containing actual data values.
        Specify this parameter if you want to compare forecasted values with
        their actual values.
    actual_col_names : dict, optional
        Dictionary mapping actual DataFrame columns, by default None.
        Keys: 'timestamp', 'value'.
    title : str, optional
        The title of the plot. If no value is provided, then
        no title is added to the plot
    plot_start_date : str, optional
        The start date for plotting data. If no value is provided,
        the plot uses all available dates in the plot.
    show_peaks : bool, default=True
        Whether to highlight peaks in the data.
    engine : str {'matplotlib', 'plotly'}, default='matplotlib'
        The plotting engine to use, either 'matplotlib' or 'plotly'.
        See the 'Notes' section for additional details.
    **plot_kwargs
        Additional keyword arguments for customization.

    Raises
    ------
    ValueError
        if the specified engine is neither 'matplotlib' nor 'plotly'.

    Notes
    -----
    The `engine` parameter allows you to specify which library gets used
    to generate the plots. Using `'engine='plotly'` generates prettier
    plots. However, it requires you to have `plotly` library installed.
    By default, `plotly` is also included in the project `requirements.txt` file.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "date": ["2023-01-01", "2023-01-02"],
    ...     "value": [10, 15]
    ... })
    >>> plot_historical_and_forecast(df,"date","value",title="Sample Plot",engine="matplotlib")

    """
    figsize = plot_kwargs.get("figsize", (20, 6))
    plot_title = plot_kwargs.get("plot_title", True)
    title_fontsize = plot_kwargs.get("title_fontsize", 16)
    plot_legend = plot_kwargs.get("plot_legend", True)
    loc = plot_kwargs.get("loc", "upper left")
    prop = plot_kwargs.get("prop", {"size": 12})

    if not isinstance(title, str):
        plot_title = False

    # Preprocess the input timeseries
    input_timeseries = convert_to_datetime(input_timeseries, timestamp_col_name)
    input_timeseries = filter_by_date(input_timeseries, timestamp_col_name,
                                      plot_start_date)
    input_timeseries = input_timeseries.sort_values(timestamp_col_name)

    forecast_col_names = forecast_col_names or {}
    forecast_timestamp = forecast_col_names.get("timestamp", "date")
    forecast_value = forecast_col_names.get("value", "forecast_value")
    confidence_level_col = forecast_col_names.get("confidence_level", "confidence_level")
    low_ci_col = forecast_col_names.get("lower_bound", "prediction_interval_lower_bound")
    upper_ci_col = forecast_col_names.get("upper_bound", "prediction_interval_upper_bound")

    if engine == "matplotlib":

        plt.figure(figsize=figsize)
        plot_series(
            input_timeseries[timestamp_col_name],
            input_timeseries[data_col_name],
            label="Historical",
        )
        plt.xlabel(timestamp_col_name)
        plt.ylabel(data_col_name)

        if show_peaks:
            peaks, _ = find_peaks(input_timeseries[data_col_name], height=0)
            peak_indices = input_timeseries.iloc[peaks].nlargest(9, data_col_name).index
            mask = input_timeseries.index.isin(peak_indices)
            plot_series(
                input_timeseries[mask][timestamp_col_name],
                input_timeseries[mask][data_col_name],
                label="Peaks",
                marker="x",
                color="red",
            )

        if forecast_output is not None:
            forecast_output = convert_to_datetime(forecast_output, forecast_timestamp)
            forecast_output = forecast_output.sort_values(forecast_timestamp)

            plot_series(
                forecast_output[forecast_timestamp],
                forecast_output[forecast_value],
                label="Forecast",
                linestyle="--",
            )

            if (
                confidence_level_col in forecast_output.columns
                and low_ci_col in forecast_output.columns
                and upper_ci_col in forecast_output.columns
            ):
                confidence_level = forecast_output[confidence_level_col].iloc[0] * 100
                plt.fill_between(
                    forecast_output[forecast_timestamp],
                    forecast_output[low_ci_col],
                    forecast_output[upper_ci_col],
                    color="#539caf",
                    alpha=0.4,
                    label=f"{confidence_level}% confidence interval",
                )
            else:
                missing_cols = [
                    col
                    for col in [confidence_level_col, low_ci_col, upper_ci_col]
                    if col not in forecast_output.columns
                ]
                print(
                    f"Warning: Missing columns {missing_cols} in forecast_output. "
                    "Continuing without confidence interval."
                )

        if actual is not None:
            actual_col_names = actual_col_names or {}
            actual_timestamp = actual_col_names.get("timestamp", timestamp_col_name)
            actual_value = actual_col_names.get("value", data_col_name)

            actual = convert_to_datetime(actual, actual_timestamp)
            actual = actual.sort_values(actual_timestamp)
            plot_series(
                actual[actual_timestamp],
                actual[actual_value],
                label="Actual",
                linestyle="--",
            )

        if plot_title:
            plt.title(title, fontsize=title_fontsize)

        if plot_legend:
            plt.legend(loc=loc, prop=prop)

        plt.show()

    elif engine == "plotly":
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=input_timeseries[timestamp_col_name],
                y=input_timeseries[data_col_name],
                mode="lines",
                name="Historical",
            )
        )

        if show_peaks:
            peaks, _ = find_peaks(input_timeseries[data_col_name], height=0)
            peak_indices = input_timeseries.iloc[peaks].nlargest(9, data_col_name).index
            mask = input_timeseries.index.isin(peak_indices)
            fig.add_trace(
                go.Scatter(
                    x=input_timeseries[mask][timestamp_col_name],
                    y=input_timeseries[mask][data_col_name],
                    mode="markers",
                    name="Peaks",
                    marker=dict(color="red", symbol="x"),
                )
            )

        if forecast_output is not None:
            forecast_output = convert_to_datetime(forecast_output, forecast_timestamp)
            forecast_output = forecast_output.sort_values(forecast_timestamp)

            fig.add_trace(
                go.Scatter(
                    x=forecast_output[forecast_timestamp],
                    y=forecast_output[forecast_value],
                    mode="lines",
                    name="Forecast",
                    line=dict(dash="dash", color="red"),
                )
            )

            if (
                confidence_level_col in forecast_output.columns
                and low_ci_col in forecast_output.columns
                and upper_ci_col in forecast_output.columns
            ):
                confidence_level = forecast_output[confidence_level_col].iloc[0] * 100
                fig.add_trace(
                    go.Scatter(
                        x=forecast_output[forecast_timestamp],
                        y=forecast_output[low_ci_col],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=forecast_output[forecast_timestamp],
                        y=forecast_output[upper_ci_col],
                        mode="lines",
                        fill="tonexty",
                        line=dict(width=0),
                        fillcolor="rgba(83, 156, 175, 0.4)",
                        name=f"{confidence_level}% confidence interval",
                    )
                )
            else:
                missing_cols = [
                    col
                    for col in [confidence_level_col, low_ci_col, upper_ci_col]
                    if col not in forecast_output.columns
                ]
                print(
                    f"Warning: Missing columns {missing_cols} in forecast_output. "
                    f"Continuing without confidence interval."
                )

        if actual is not None:
            actual_col_names = actual_col_names or {}
            actual_timestamp = actual_col_names.get("timestamp", timestamp_col_name)
            actual_value = actual_col_names.get("value", data_col_name)

            actual = convert_to_datetime(actual, actual_timestamp)
            actual = actual.sort_values(actual_timestamp)
            fig.add_trace(
                go.Scatter(
                    x=actual[actual_timestamp],
                    y=actual[actual_value],
                    mode="lines",
                    name="Actual",
                    line=dict(color="green"),
                )
            )

        if plot_title:
            fig.update_layout(title=dict(text=title, font=dict(size=title_fontsize)))

        if plot_legend:
            fig.update_layout(legend=dict(x=0, y=1.0, font=dict(size=prop["size"])))

        fig.update_layout(height=400)
        fig.show()
    else:
        raise ValueError("Engine must be either 'matplotlib' or 'plotly'")
