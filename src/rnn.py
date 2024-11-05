import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Assuming you have a custom dataset for your time series data
class TimeSeriesDataset(Dataset):
    def __init__(self, series, input_size, output_size):
        """
        TimeSeriesDataset for preparing time series data for RNN training.

        Parameters:
        - series: Input time series data.
        - input_size: Number of input time steps (lags).
        - output_size: Number of time steps to forecast (forecast horizon).
        """
        self.series = series
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return len(self.series) - self.input_size - self.output_size

    def __getitem__(self, idx):
        """
        Get the input and output sequence for the model.

        Parameters:
        - idx: Index to slice the data.

        Returns:
        - input_seq: Input time steps.
        - output_seq: Target forecast horizon.
        """
        input_seq = self.series[idx:idx + self.input_size]
        output_seq = self.series[idx + self.input_size:idx + self.input_size + self.output_size]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(output_seq, dtype=torch.float32)

class Seq2SeqRNN(nn.Module):
    """
    Sequence-to-sequence RNN model for time series forecasting.
    """
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1, output_seq_len=12):
        super(Seq2SeqRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = output_seq_len

        # Encoder RNN
        self.encoder = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Decoder RNN
        self.decoder = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer to map hidden state to output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        """
        Forward pass for the Seq2Seq RNN model.

        Parameters:
        - x: Input sequence tensor (batch_size, seq_len, input_size).
        - target: Target sequence for teacher forcing (optional).
        - teacher_forcing_ratio: Probability to use teacher forcing during training.

        Returns:
        - outputs: Predicted output sequence (batch_size, output_seq_len, output_size).
        """
        batch_size = x.size(0)

        # Encode the input sequence
        encoder_output, hidden = self.encoder(x)

        # Initialize decoder input with the last value of encoder input
        decoder_input = x[:, -1, :].unsqueeze(1)  # Shape: (batch_size, 1, input_size)

        outputs = []
        for t in range(self.output_seq_len):
            # Decode step
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            output = self.fc(decoder_output)
            outputs.append(output)

            # Teacher forcing: feed the target as the next input
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t, :].unsqueeze(1)  # Corrected line
            else:
                decoder_input = output

        outputs = torch.cat(outputs, dim=1)  # Shape: (batch_size, output_seq_len, output_size)
        return outputs.squeeze(-1)  # Remove last dimension
    
def train_model(train_series, num_input_lags, forecast_horizon, hidden_size, num_layers, batch_size=32, num_epochs=10, learning_rate=0.001):
    """
    Train the RNN model on the training data.

    Parameters:
    - train_series: Time series data for training (1D numpy array).
    - num_input_lags: Number of input time steps (lags) used as model input.
    - forecast_horizon: Number of time steps to forecast.
    - batch_size: Batch size for training.
    - num_epochs: Number of training epochs.
    - learning_rate: Learning rate for the optimizer.

    Returns:
    - model: Trained RNN model.
    """
    # Create dataset and dataloader
    dataset = TimeSeriesDataset(train_series, num_input_lags, forecast_horizon)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = Seq2SeqRNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1, output_seq_len=forecast_horizon)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            # Reshape inputs and targets
            x_batch = x_batch.unsqueeze(-1)  # Shape: (batch_size, seq_len, input_size)
            y_batch = y_batch.unsqueeze(-1)  # Shape: (batch_size, seq_len, input_size)

            optimizer.zero_grad()
            output = model(x_batch, target=y_batch, teacher_forcing_ratio=0.5)
            loss = criterion(output, y_batch.squeeze(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        #print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

    return model

def generate_predictions(model, input_sequence):
    """
    Generate predictions using the trained model.

    Parameters:
    - model: Trained RNN model.
    - input_sequence: Input sequence for prediction (last 'num_input_lags' points, 1D array).

    Returns:
    - predicted_original_scale: Predictions in the original scale (after inverse log transformation).
    """
    model.eval()
    with torch.no_grad():
        # Prepare the input sequence
        x_input = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # Shape: (1, seq_len, input_size)

        # Predict using RNN
        output = model(x_input)
        predicted = output.squeeze(0).numpy()

    return predicted

def forecast_with_rnn(last_month, **kwargs):
    """
    Forecast using RNN model up to the given last_month.

    Parameters:
    - last_month (Timestamp): The last month to consider in the time series.
    - kwargs: Additional arguments needed for forecasting.

    Returns:
    - np.ndarray: Predicted values in the transformed scale (e.g., differenced log).
    """
    # Unpack necessary arguments
    diff_log_series = kwargs['diff_log_series']
    num_input_lags = kwargs['num_input_lags']
    num_epochs = kwargs['num_epochs']
    learning_rate = kwargs['learning_rate']
    forecast_horizon = kwargs['forecast_horizon']
    batch_size = kwargs['batch_size']
    num_layers = kwargs['num_layers']
    hidden_size = kwargs['hidden_size']

    # Slice the series up to last_month
    series_temp = diff_log_series.loc[:last_month]

    # Ensure we have enough data points
    if len(series_temp) < num_input_lags:
        raise ValueError("Not enough data to create input sequence for RNN.")

    # Train RNN model
    model = train_model(
        series_temp.values, 
        num_input_lags, 
        forecast_horizon, 
        batch_size=batch_size, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        num_layers=num_layers
    )

    # Prepare input sequence
    input_sequence = series_temp.values[-num_input_lags:]

    # Generate RNN predictions in the transformed scale
    rnn_predictions = generate_predictions(model, input_sequence)

    return rnn_predictions