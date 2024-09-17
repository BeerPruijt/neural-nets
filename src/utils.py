import numpy as np
import torch 
import pandas as pd
from torch.utils.data import Dataset

def preprocess_data(series):
    """
    Preprocess the data by taking logarithm and differencing.

    Parameters:
    - series: numpy array of raw data

    Returns:
    - diff_log_series: differenced log-transformed series
    - log_series: log-transformed series
    """
    # Preprocessing: Log-differencing
    log_series = np.log(series)
    diff_log_series = np.diff(log_series)
    return diff_log_series, log_series


class TimeSeriesDataset(Dataset):
    """
    Custom Dataset class for time series data.
    """
    def __init__(self, series, input_window, output_window):
        self.series = series
        self.input_window = input_window
        self.output_window = output_window

    def __len__(self):
        return len(self.series) - self.input_window - self.output_window + 1

    def __getitem__(self, idx):
        x = self.series[idx:idx + self.input_window]
        y = self.series[idx + self.input_window:idx + self.input_window + self.output_window]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
def generate_plot(rnn_predicted, ar_predicted, data, forecast_horizon, num_input_lags):
    """
    Plot the last 12 observations from the data and append the RNN and AR model predictions.

    Parameters:
    - rnn_predicted: RNN model predictions in the original scale
    - ar_predicted: AR model predictions in the original scale
    - data: Original dataset
    - forecast_horizon: Number of time steps ahead for forecasts
    - num_input_lags: Number of input lags for the AR model
    """
    import matplotlib.pyplot as plt
    
    # Get the last 12 observations from the data
    last_12_obs = data[-12:]
    
    # Create the x-axis for the actual and forecasted values
    historical_time_steps = range(-11, 1)  # Last 12 observations (counting backwards from -11 to 0)
    forecast_time_steps = range(1, forecast_horizon + 1)  # Forecasted values (1 to forecast_horizon)
    
    plt.figure(figsize=(12, 6))
    
    # Plot the last 12 observations from the data
    plt.plot(historical_time_steps, last_12_obs, label='Last 12 Observations', marker='o')
    
    # Plot RNN model predictions
    plt.plot(forecast_time_steps, rnn_predicted, label='RNN Predicted', marker='o')
    
    # Plot AR model predictions
    plt.plot(forecast_time_steps, ar_predicted, label=f'AR({num_input_lags}) Predicted', marker='o')
    
    # Add titles and labels
    plt.title('Last 12 Observations and Forecasted Inflation Index')
    plt.xlabel('Time Steps')
    plt.ylabel('Inflation Index')
    
    # Add legend
    plt.legend()
    
    # Display the plot
    plt.show()

def get_train_test_idxs(data, column, train_perc, forecast_horizon):
    valid_idxs = data[column].dropna().index[0:-forecast_horizon]
    n_train = round(len(valid_idxs)*train_perc)
    train_idxs = valid_idxs[0:n_train]
    test_idxs = valid_idxs[n_train:]
    return valid_idxs, train_idxs, test_idxs

def construct_storage_df(test_idxs, forecast_horizon):
    return pd.DataFrame(np.nan, index=test_idxs, columns=[f't_{i}' for i in range(1, forecast_horizon+1)])

def forecast_with_observations(last_month, **kwargs):
    """
    Use actual observations as a naive forecasting method.

    Parameters:
    - last_month (Timestamp): The last month to consider in the time series.
    - kwargs: Additional arguments needed for forecasting.

    Returns:
    - np.ndarray: Observed values from the data.
    """
    data = kwargs['data']
    # Generate forecast months
    start_month = pd.to_datetime(last_month) + pd.DateOffset(months=1)
    forecast_months = pd.date_range(start=start_month, periods=12, freq='MS')
    observations = data.loc[forecast_months, 'C000000'].values
    return observations

def run_forecasts(storage_df, forecasting_method, **kwargs):
    """
    Run forecasts using the specified forecasting method and store results.

    Parameters:
    - storage_df (DataFrame): DataFrame to store forecast results.
    - forecasting_method (function): The forecasting function to use.
    - kwargs: Additional arguments passed to the forecasting method.

    Returns:
    - DataFrame: The updated storage DataFrame with forecasts.
    """
    for last_month in storage_df.index:
        print(f"Processing forecasts for: {last_month}")

        # Generate predictions
        predictions = forecasting_method(last_month, **kwargs)
        # Store the predictions
        storage_df.loc[last_month] = predictions

    return storage_df

def log_diff_column(dataframe, column, dropna=True):
    series = dataframe.loc[:, column]
    series = np.log(series).diff()
    if dropna:
        series = series.dropna()
    return series

def validate_inputs_for_inversion(log_diffed_series, level_dataframe):
    """
    Validates inputs for inverting a log-differenced series back to its original scale.

    Parameters
    ----------
    log_diffed_series : pandas.Series
        A Series containing log-differenced values. Must have a DateTimeIndex.
    level_dataframe : pandas.DataFrame
        A DataFrame containing the original level values. Must have a DateTimeIndex.

    Raises
    ------
    TypeError
        If log_diffed_series is not a pandas Series.
        If the indices of log_diffed_series or level_dataframe are not DateTimeIndex.
    ValueError
        If the name of log_diffed_series is not in level_dataframe columns.
        If the index frequencies of log_diffed_series and level_dataframe do not match.
        If the index before the first index in log_diffed_series is not present in level_dataframe index.
        If the initial value in level_dataframe is NaN.
    """
    # Check that log_diffed_series is a pandas Series
    if not isinstance(log_diffed_series, pd.Series):
        raise TypeError("log_diffed_series must be a pandas Series.")

    # Check that the indices are DateTimeIndex
    if not isinstance(log_diffed_series.index, pd.DatetimeIndex):
        raise TypeError("The index of log_diffed_series must be a pandas DateTimeIndex.")
    if not isinstance(level_dataframe.index, pd.DatetimeIndex):
        raise TypeError("The index of level_dataframe must be a pandas DateTimeIndex.")

    # Check that log_diffed_series has a name that is present in level_dataframe columns
    series_name = log_diffed_series.name
    if series_name not in level_dataframe.columns:
        raise ValueError(f"The name of log_diffed_series '{series_name}' is not present in level_dataframe columns.")

    # Check that log_diffed_series index has the same frequency as level_dataframe
    log_diff_freq = pd.infer_freq(log_diffed_series.index)
    level_df_freq = pd.infer_freq(level_dataframe.index)
    if log_diff_freq != level_df_freq:
        raise ValueError(
            f"The index frequency of log_diffed_series ({log_diff_freq}) does not match that of level_dataframe ({level_df_freq})."
        )

    # Get the position of the first index in level_dataframe
    first_index = log_diffed_series.index[0]
    try:
        pos_in_level_df = level_dataframe.index.get_loc(first_index)
    except KeyError:
        raise ValueError(
            f"The first index of log_diffed_series ({first_index}) is not present in level_dataframe index."
        )

    if pos_in_level_df == 0:
        raise ValueError(
            f"The index before the first index in log_diffed_series ({first_index}) is not available in level_dataframe."
        )

    # Get the index before the first index in log_diffed_series
    prev_index = level_dataframe.index[pos_in_level_df - 1]

    # Ensure that prev_index is in level_dataframe index
    if prev_index not in level_dataframe.index:
        raise ValueError(
            f"The index before the first index in log_diffed_series ({prev_index}) is not present in level_dataframe index."
        )

    # Get the initial value from level_dataframe
    initial_value = level_dataframe.loc[prev_index, series_name]
    if pd.isna(initial_value):
        raise ValueError(f"The initial value at index {prev_index} is NaN in level_dataframe.")

    # Get the initial value from level_dataframe
    if initial_value == 0:
        raise ValueError(f"The initial value at index {prev_index} is 0 in level_dataframe.")

    # Get the initial value from level_dataframe
    if initial_value < 0:
        raise ValueError(f"The initial value at index {prev_index} is negative in level_dataframe.")

    # Check for NaN values in log_diffed_series
    if log_diffed_series.isnull().any():
        raise ValueError("log_diffed_series contains NaN values")

    return initial_value, pos_in_level_df


def invert_log_diff_series(log_diffed_series, level_dataframe):
    """
    Reconstructs the original time series from log-differenced values using the appropriate initial value.

    Parameters
    ----------
    log_diffed_series : pandas.Series
        A Series containing log-differenced values. The Series must have a name that is present in level_dataframe columns.
    level_dataframe : pandas.DataFrame
        A DataFrame containing the original level values. Must contain the initial value needed for reconstruction.

    Returns
    -------
    pandas.Series
        A Series containing the reconstructed values in the original scale.

    Raises
    ------
    TypeError
        If log_diffed_series is not a pandas Series.
    ValueError
        If the name of log_diffed_series is not in level_dataframe columns.
        If the index frequencies of log_diffed_series and level_dataframe do not match.
        If the index before the first index in log_diffed_series is not present in level_dataframe index.
        If the initial value in level_dataframe is NaN.
    """
    # Validate inputs and get initial value
    initial_value, _ = validate_inputs_for_inversion(log_diffed_series, level_dataframe)

    # Get the log of the initial value
    log_initial_value = np.log(initial_value)

    # Reconstruct the log levels by cumulatively summing the log differences
    log_levels = log_diffed_series.cumsum() + log_initial_value

    # Exponentiate to get back to the original scale
    reconstructed_series = np.exp(log_levels)
    reconstructed_series.name = log_diffed_series.name

    return reconstructed_series