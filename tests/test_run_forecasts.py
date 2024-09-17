
import pytest
import pandas as pd
import numpy as np
from src.utils import run_forecasts

def dummy_forecasting_method(last_month, **kwargs):
    """
    Dummy forecasting method for testing purposes.
    Returns an array of ones with length equal to forecast_horizon.
    """
    forecast_horizon = kwargs.get('forecast_horizon', 3)
    return np.ones(forecast_horizon)

@pytest.fixture
def storage_df():
    """
    Fixture to create a storage DataFrame for testing.
    """
    index = pd.date_range(start='2020-01-01', periods=5, freq='M')
    forecast_horizon = 3  # Number of periods to forecast
    columns = range(forecast_horizon)
    # Initialize the DataFrame with dtype=float
    df = pd.DataFrame(index=index, columns=columns, dtype=float)
    return df

def test_run_forecasts(storage_df):
    """
    Test the run_forecasts function to ensure it works as intended.
    """
    # Define kwargs for forecasting_method
    forecast_horizon = 3
    kwargs = {
        'forecast_horizon': forecast_horizon
    }

    # Run the forecasts
    result_df = run_forecasts(storage_df.copy(), dummy_forecasting_method, **kwargs)

    # Expected result: DataFrame filled with ones
    expected_df = pd.DataFrame(
        data=np.ones((len(storage_df.index), forecast_horizon)),
        index=storage_df.index,
        columns=storage_df.columns
    )

    # Assert that the result DataFrame matches the expected DataFrame
    pd.testing.assert_frame_equal(result_df, expected_df)

    print("run_forecasts function passed the test with dummy_forecasting_method.")