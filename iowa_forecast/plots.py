import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import find_peaks


def convert_to_datetime(df, col):
    df[col] = pd.to_datetime(df[col])
    return df


def filter_by_date(df, col, start_date):
    if start_date:
        return df[df[col] >= pd.to_datetime(start_date)]
    return df


def plot_historical_and_forecast(
    input_timeseries,
    timestamp_col_name,
    data_col_name,
    forecast_output=None,
    forecast_col_names=None,
    actual=None,
    actual_col_names=None,
    title=None,
    plotstartdate=None,
    show_peaks=True,
    engine="matplotlib",
    **plot_kwargs,
):
    figsize = plot_kwargs.get("figsize", (20, 6))
    plot_title = plot_kwargs.get("plot_title", True)
    title_fontsize = plot_kwargs.get("title_fontsize", 16)
    plot_legend = plot_kwargs.get("plot_legend", True)
    loc = plot_kwargs.get("loc", "upper left")
    prop = plot_kwargs.get("prop", {"size": 12})

    # Preprocess the input timeseries
    input_timeseries = convert_to_datetime(input_timeseries, timestamp_col_name)
    input_timeseries = filter_by_date(input_timeseries, timestamp_col_name, plotstartdate)
    input_timeseries = input_timeseries.sort_values(timestamp_col_name)

    forecast_col_names = forecast_col_names or {}
    forecast_timestamp = forecast_col_names.get("timestamp", "date")
    forecast_value = forecast_col_names.get("value", "forecast_value")
    confidence_level_col = forecast_col_names.get("confidence_level", "confidence_level")
    low_CI_col = forecast_col_names.get("lower_bound", "prediction_interval_lower_bound")
    upper_CI_col = forecast_col_names.get("upper_bound", "prediction_interval_upper_bound")

    if engine == "matplotlib":

        def plot_series(x, y, label, linestyle="-", marker=None, color=None):
            plt.plot(x, y, label=label, linestyle=linestyle)
            if marker:
                plt.scatter(x, y, marker=marker, color=color)

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
                and low_CI_col in forecast_output.columns
                and upper_CI_col in forecast_output.columns
            ):
                confidence_level = forecast_output[confidence_level_col].iloc[0] * 100
                plt.fill_between(
                    forecast_output[forecast_timestamp],
                    forecast_output[low_CI_col],
                    forecast_output[upper_CI_col],
                    color="#539caf",
                    alpha=0.4,
                    label=f"{confidence_level}% confidence interval",
                )
            else:
                missing_cols = [
                    col
                    for col in [confidence_level_col, low_CI_col, upper_CI_col]
                    if col not in forecast_output.columns
                ]
                print(
                    f"Warning: Missing columns {missing_cols} in forecast_output. Continuing without confidence interval."
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
                and low_CI_col in forecast_output.columns
                and upper_CI_col in forecast_output.columns
            ):
                confidence_level = forecast_output[confidence_level_col].iloc[0] * 100
                fig.add_trace(
                    go.Scatter(
                        x=forecast_output[forecast_timestamp],
                        y=forecast_output[low_CI_col],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=forecast_output[forecast_timestamp],
                        y=forecast_output[upper_CI_col],
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
                    for col in [confidence_level_col, low_CI_col, upper_CI_col]
                    if col not in forecast_output.columns
                ]
                print(
                    f"Warning: Missing columns {missing_cols} in forecast_output. Continuing without confidence interval."
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
