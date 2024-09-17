import pytest
import pandas as pd
import numpy as np

# Replace 'your_module' with the actual module name where the function is defined
from src.utils import log_diff_column

def test_basic_functionality():
    """Test the function with a simple increasing sequence."""
    data = pd.DataFrame({'A': [1, 2, 4, 8, 16]})
    expected_values = np.log(data['A']).diff().dropna()
    expected = pd.Series(expected_values.values, index=[1, 2, 3, 4], name='A')
    result = log_diff_column(data, 'A')
    pd.testing.assert_series_equal(result, expected)

def test_dropna_false():
    """Test the function with dropna set to False."""
    data = pd.DataFrame({'A': [1, 2, 4, 8, 16]})
    expected_values = np.log(data['A']).diff()
    expected = pd.Series(expected_values.values, index=[0, 1, 2, 3, 4], name='A')
    result = log_diff_column(data, 'A', dropna=False)
    pd.testing.assert_series_equal(result, expected)

def test_with_zeros():
    """Test the function when the data contains zeros (log(0) is undefined)."""
    data = pd.DataFrame({'A': [1, 0, 4, 0, 16]})
    with pytest.warns(RuntimeWarning):
        result = log_diff_column(data, 'A', dropna=False)
        expected = np.log(data['A']).diff()
    
    pd.testing.assert_series_equal(result, expected)

def test_with_nan_values():
    """Test the function when the data contains NaN values."""
    data = pd.DataFrame({'A': [1, np.nan, 4, np.nan, 16]})
    result = log_diff_column(data, 'A', dropna=False)
    expected = np.log(data['A']).diff()
    pd.testing.assert_series_equal(result, expected)

def test_nonexistent_column():
    """Test the function when the specified column does not exist."""
    data = pd.DataFrame({'A': [1, 2, 3]})
    with pytest.raises(KeyError):
        log_diff_column(data, 'B')

def test_non_numeric_data():
    """Test the function when the column contains non-numeric data."""
    data = pd.DataFrame({'A': ['a', 'b', 'c']})
    with pytest.raises(TypeError):
        log_diff_column(data, 'A')

def test_empty_dataframe():
    """Test the function when the DataFrame is empty."""
    data = pd.DataFrame({'A': []})
    result = log_diff_column(data, 'A')
    expected = pd.Series([], name='A', dtype=float)
    pd.testing.assert_series_equal(result, expected)
