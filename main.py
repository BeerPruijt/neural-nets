from src.utils import (
    log_diff_column,
    construct_storage_df,
    get_train_test_idxs,
    run_forecasts,
    forecast_with_observations,
)
from src.rnn import forecast_with_rnn
import numpy as np
import pandas as pd
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    # Hyperparameters to optimize
    num_input_lags = trial.suggest_int('num_input_lags', 1, 12)
    batch_size = trial.suggest_categorical('batch_size', [1, 8, 16, 32])
    num_epochs = trial.suggest_categorical('num_epochs', [10, 20, 40, 80, 160])
    hidden_size = trial.suggest_int('hidden_size', 10, 100)
    num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
    learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-1, log=True)
    
    # Settings
    LOC = r"G:\EBO\ECMO\NIPE\NIPE_automated\Output\2024-Q4\11-09-2024 (prism_check)\Results\df_original.xlsx"
    COLUMN = 'C000000'
    TRAIN_PERC = 0.8
    FORECAST_HORIZON = 12  # Keeping this fixed
    
    # Load your univariate time series data
    data = pd.read_excel(LOC, parse_dates=True, index_col=0)
    
    # Prepare indices
    valid_idxs, _, test_idxs = get_train_test_idxs(data, COLUMN, TRAIN_PERC, FORECAST_HORIZON)
    
    # Preprocess data
    ld_series = log_diff_column(data, COLUMN)
    
    # Initialize storage DataFrames
    df_observations = construct_storage_df(test_idxs, FORECAST_HORIZON)
    df_rnn = construct_storage_df(test_idxs, FORECAST_HORIZON)
    
    # Prepare kwargs for the forecasting methods
    rnn_kwargs = {
        'diff_log_series': ld_series,
        'num_input_lags': num_input_lags,
        'num_epochs': num_epochs,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'forecast_horizon': FORECAST_HORIZON,
        'learning_rate': learning_rate,
        'batch_size': batch_size
    }
    
    observations_kwargs = {
    'data': data,
    'how': 'dlog',
    'forecast_horizon': FORECAST_HORIZON,
    'column': COLUMN
    }
    
    # Run the forecasts using RNN
    df_rnn = run_forecasts(df_rnn, forecast_with_rnn, **rnn_kwargs)
    
    # Run the forecasts using observations
    df_observations = run_forecasts(df_observations, forecast_with_observations, **observations_kwargs)
    
    # Compute the MSE between df_rnn and df_observations
    mse = ((df_rnn - df_observations) ** 2).mean().mean()
        
    return mse

if __name__ == '__main__':

    EXPERIMENT_NAME = 'rnn_hyperparameter_optimization'

    # Set up MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Set up MLflowCallback for Optuna
    mlflow_callback = MLflowCallback(
        tracking_uri=None,
        metric_name='mse',
        create_experiment=False    
    )
    
    # Create Optuna study
    study = optuna.create_study(direction='minimize')
    
    # Optimize the objective function
    study.optimize(objective, n_trials=100, callbacks=[mlflow_callback])
    
    # Print best trial results
    print('Best trial:')
    trial = study.best_trial
    
    print(f'  Value (MSE): {trial.value}')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
