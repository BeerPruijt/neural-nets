import numpy as np
from statsmodels.tsa.ar_model import AutoReg

def forecast_with_ar(last_month, **kwargs):
    """
    Forecast using AR (Autoregressive) model up to the given last_month.

    Parameters:
    - last_month (Timestamp): The last month to consider in the time series.
    - kwargs: Additional arguments needed for forecasting.

    Returns:
    - np.ndarray: Predicted values in the transformed scale (e.g., differenced log).
    """
    # Unpack necessary arguments
    diff_log_series = kwargs['diff_log_series']
    num_input_lags = kwargs['num_input_lags']
    forecast_horizon = kwargs['forecast_horizon']

    # Slice the series up to last_month
    series_temp = diff_log_series.loc[:last_month]

    # Ensure we have enough data points
    if len(series_temp) < num_input_lags:
        raise ValueError("Not enough data to create input sequence for AR model.")

    # Fit AR model
    ar_model = AutoReg(series_temp.values, lags=num_input_lags)
    ar_fitted = ar_model.fit()

    # Generate AR predictions directly for forecast_horizon steps
    ar_predictions = ar_fitted.forecast(steps=forecast_horizon)

    return ar_predictions
