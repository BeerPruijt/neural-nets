import pytest
import pandas as pd
import numpy as np

from src.utils import invert_log_diff_series

def test_successful_inversion():
    """Test successful inversion with valid data."""
    # Original level data
    dates = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    level_values = pd.DataFrame({'A': np.exp(np.random.randn(5).cumsum())}, index=dates)
    
    # Compute log-differenced series
    log_diffed = np.log(level_values['A']).diff().dropna()
    log_diffed.name = 'A'
    
    # Invert the log-differenced series
    reconstructed_series = invert_log_diff_series(log_diffed, level_values)
    
    # Expected original series
    expected_series = level_values['A'].loc[log_diffed.index]
    
    # Assert that the reconstructed series matches the original series
    pd.testing.assert_series_equal(reconstructed_series, expected_series)

def test_mismatched_frequencies():
    """Test inversion when the index frequencies do not match."""
    # Original level data with monthly frequency
    dates_level = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    level_values = pd.DataFrame({'A': np.exp(np.random.randn(5).cumsum())}, index=dates_level)
    
    # Log-differenced series with daily frequency
    dates_log_diff = pd.date_range(start='2020-01-02', periods=4, freq='D')
    log_diffed = pd.Series(np.random.randn(4), index=dates_log_diff, name='A')
    
    # Attempt inversion and expect a ValueError
    with pytest.raises(ValueError, match="index frequency.*does not match"):
        invert_log_diff_series(log_diffed, level_values)

def test_missing_initial_value():
    """Test inversion when the initial value is missing in level_dataframe."""
    # Original level data
    dates = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    level_values = pd.DataFrame({'A': [np.nan, 2, 3, 4, 5]}, index=dates)
    
    # Compute log-differenced series starting from the second index
    log_diffed = pd.Series([0.6931, 0.4055, 0.2877, 0.2231], index=dates[1:], name='A')

    # Attempt inversion and expect a ValueError due to NaN initial value
    with pytest.raises(ValueError, match="initial value.*is NaN"):
        invert_log_diff_series(log_diffed, level_values)

def test_name_mismatch():
    """Test inversion when the name of log_diffed_series is not in level_dataframe columns."""
    # Original level data
    dates = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    level_values = pd.DataFrame({'A': [1, 2, 3, 4, 5]}, index=dates)
    
    # Log-differenced series with a different name
    log_diffed = pd.Series([0.6931, 0.4055, 0.2877, 0.69], index=dates[1:], name='B')
    
    # Attempt inversion and expect a ValueError due to name mismatch
    with pytest.raises(ValueError, match="name of log_diffed_series.*not present"):
        invert_log_diff_series(log_diffed, level_values)

def test_non_series_input():
    """Test inversion when log_diffed_series is not a pandas Series."""
    # Original level data
    dates = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    level_values = pd.DataFrame({'A': [1, 2, 3, 4, 5]}, index=dates)
    
    # Log-differenced data as a NumPy array instead of a Series
    log_diffed = np.array([0.6931, 0.4055, 0.2877])
    
    # Attempt inversion and expect a TypeError
    with pytest.raises(TypeError, match="log_diffed_series must be a pandas Series"):
        invert_log_diff_series(log_diffed, level_values)

def test_missing_index_in_level_dataframe():
    """Test inversion when the index before the first index in log_diffed_series is missing."""
    # Original level data missing the initial index
    dates_level = pd.date_range(start='2020-01-02', periods=5, freq='MS')
    level_values = pd.DataFrame({'A': [2, 3, 4, 5, 6]}, index=dates_level)
    
    # Log-differenced series
    dates_log_diff = pd.date_range(start='2020-01-03', periods=4, freq='MS')
    log_diffed = pd.Series([0.4055, 0.2877, 0.2231, 0.1823], index=dates_log_diff, name='A')
    
    # Attempt inversion and expect a ValueError
    with pytest.raises(ValueError, match="index before the first index.*not available"):
        invert_log_diff_series(log_diffed, level_values)

def test_non_datetime_index():
    """Test inversion when the indices are not DateTimeIndex."""
    # Original level data with integer index
    level_values = pd.DataFrame({'A': [1, 2, 3, 4, 5]}, index=[0, 1, 2, 3, 4])
    # Log-differenced series with integer index
    log_diffed = pd.Series([0.6931, 0.4055, 0.2877, 0.2231], index=[1, 2, 3, 4], name='A')
    
    # Attempt inversion and expect a TypeError
    with pytest.raises(TypeError, match="index of log_diffed_series must be a pandas DateTimeIndex"):
        invert_log_diff_series(log_diffed, level_values)

def test_inconsistent_indices():
    """Test inversion when log_diffed_series indices are not in level_dataframe."""
    # Original level data
    dates_level = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    level_values = pd.DataFrame({'A': [1, 2, 3, 4, 5]}, index=dates_level)
    
    # Log-differenced series with non-matching indices
    dates_log_diff = pd.date_range(start='2020-06-01', periods=4, freq='MS')
    log_diffed = pd.Series([0.6931, 0.4055, 0.2877, 0.2231], index=dates_log_diff, name='A')
    
    # Attempt inversion and expect a ValueError
    with pytest.raises(ValueError, match="first index of log_diffed_series.*not present"):
        invert_log_diff_series(log_diffed, level_values)

def test_inversion_with_nan_in_log_diffed_series():
    """Test inversion when log_diffed_series contains NaN values."""
    # Original level data
    dates = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    level_values = pd.DataFrame({'A': np.exp(np.random.randn(5).cumsum())}, index=dates)

    # Compute log-differenced series and drop the initial NaN
    log_diffed = np.log(level_values['A']).diff().dropna()
    log_diffed.name = 'A'

    # Introduce a NaN value in log_diffed_series
    log_diffed.iloc[1] = np.nan  # Introduce NaN at the second position

    # Attempt inversion and expect a ValueError due to NaN in log_diffed_series
    with pytest.raises(ValueError, match="log_diffed_series contains NaN values"):
        invert_log_diff_series(log_diffed, level_values)

def test_inversion_with_zero_initial_value():
    """Test inversion when the initial value in level_dataframe is zero."""
    # Original level data with zero initial value
    dates = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    level_values = pd.DataFrame({'A': [0, 2, 3, 4, 5]}, index=dates)
    
    # Compute log-differenced series starting from the second index
    log_diffed =  pd.Series([0.4055, 0.2877, 0.2231, 0.96], index=dates[1:], name='A')
    
    # Attempt inversion and expect a RuntimeWarning due to log(0)
    with pytest.raises(ValueError, match="initial value.*is 0"):
        invert_log_diff_series(log_diffed, level_values)

def test_inversion_with_negative_initial_value():
    """Test inversion when the initial value in level_dataframe is negative."""
    # Original level data with negative initial value
    dates = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    level_values = pd.DataFrame({'A': [-1, 2, 3, 4, 5]}, index=dates)
    
    # Compute log-differenced series starting from the second index
    log_diffed =  pd.Series([0.4055, 0.2877, 0.2231, 0.96], index=dates[1:], name='A')
    
    # Attempt inversion and expect a ValueError due to invalid value in log
    with pytest.raises(ValueError, match="initial value.*is negative"):
        invert_log_diff_series(log_diffed, level_values)
