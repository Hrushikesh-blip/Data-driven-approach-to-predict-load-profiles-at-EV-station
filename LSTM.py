import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from torch.utils.data import DataLoader, TensorDataset

def reconstruct_time_series(sequences, sequence_length):

    total_length = len(sequences) + sequence_length - 1
    num_features = sequences.shape[-1]
    
    # Initialize arrays for reconstruction
    full_series = np.zeros((total_length, num_features))
    counts = np.zeros((total_length, num_features))

    for i in range(len(sequences)):
        end_idx = i + sequence_length
        overlap_length = min(sequence_length, total_length - i)  # Handle edge cases
        
        # Add the sequence to the full series
        full_series[i:end_idx, :] += sequences[i, :overlap_length]
        counts[i:end_idx, :] += 1  # Track the counts for averaging

    # Avoid division by zero
    counts[counts == 0] = 1
    reconstructed_series = full_series / counts
    return reconstructed_series
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, prediction_length, dropout_prob=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_length = prediction_length
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size * prediction_length)  # Predict all timesteps at once

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))  # LSTM outputs
        out = out[:, -1, :]  # Take the last hidden state
        out = self.fc(out)  # Fully connected layer
        out = out.view(-1, self.prediction_length, int(out.size(-1) / self.prediction_length))  # Reshape to (batch, pred_len, output_size)
        return out

# Load and preprocess data
file_path = '/home/mant_hr/project_x/Final presentation/EV_station_data.csv'
df = pd.read_csv(file_path)

# Filter for Tower 98
tower_id = 98
df_tower_98 = df[df['tower_id'] == tower_id]
df_tower_98 = df_tower_98.pivot(index='timestamp', columns='tower_id', values='residual')

# Scale data
scaler = MinMaxScaler()
df_tower_98_scaled = scaler.fit_transform(df_tower_98)

# Sequence creation
sequence_length = 24
prediction_length = 4
lead_time = 0

def create_sequences(data, sequence_length, prediction_length, lead_time):
    sequences, targets = [], []
    for i in range(len(data) - sequence_length - prediction_length - lead_time):
        sequence = data[i:i + sequence_length]
        target = data[i + sequence_length + lead_time:i + sequence_length + lead_time + prediction_length]
        sequences.append(sequence)
        targets.append(target)
    return np.array(sequences), np.array(targets)

X, y = create_sequences(df_tower_98_scaled, sequence_length, prediction_length, lead_time)
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Split data into train, validation, and test sets
train_size = int(len(X) * 0.64)
val_size = int(len(X) * 0.16)
test_size = len(X) - train_size - val_size

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

# Convert to PyTorch tensors
train_tensor_x = torch.tensor(X_train, dtype=torch.float32)
train_tensor_y = torch.tensor(y_train, dtype=torch.float32)
val_tensor_x = torch.tensor(X_val, dtype=torch.float32)
val_tensor_y = torch.tensor(y_val, dtype=torch.float32)
test_tensor_x = torch.tensor(X_test, dtype=torch.float32)
test_tensor_y = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoaders
batch_size = 8
train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
val_dataset = TensorDataset(val_tensor_x, val_tensor_y)
test_dataset = TensorDataset(test_tensor_x, test_tensor_y)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model parameters
input_size = 1  # Single feature
hidden_size = 128
output_size = 1  # Single output
num_layers = 2

model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, prediction_length=prediction_length)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training loop with early stopping
epochs = 500
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 10  # Early stopping patience
trigger_times = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            val_outputs = model(batch_x)
            val_loss += criterion(val_outputs, batch_y).item()

    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Plot loss curves
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Train vs Validation Loss")
plt.show()

# Test evaluation
model.eval()
test_outputs = []
test_targets = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        output = model(batch_x)
        test_outputs.append(output.cpu().numpy())
        test_targets.append(batch_y.cpu().numpy())

test_outputs = np.concatenate(test_outputs, axis=0)
test_targets = np.concatenate(test_targets, axis=0)

test_outputs = reconstruct_time_series(test_outputs, prediction_length)
test_targets = reconstruct_time_series(test_targets, prediction_length)

# Inverse scaling
predictions = scaler.inverse_transform(test_outputs)
actual_values = scaler.inverse_transform(test_targets)

mse = mean_squared_error(actual_values, predictions)
rmse = sqrt(mse)
mae = mean_absolute_error(actual_values, predictions)
r2 = r2_score(actual_values, predictions)

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
# Plot predictions vs actuals
timestamps = df_tower_98.index[-6:]
timestamps = pd.to_datetime(timestamps)
last_hours = min(6, len(timestamps))

plt.figure(figsize=(10, 6))
plt.plot(timestamps[-last_hours:], actual_values, label="Actual", color='b', marker='o')
plt.plot(timestamps[-last_hours:], predictions, label="Predicted", color='r', marker='o')
plt.xlabel("Timestamps")
plt.ylabel("kWh")
plt.xticks(rotation=45)
plt.legend()
plt.title(f"Actual vs Predicted kWh for Tower ID {tower_id} (Last {last_hours} Hours)")
plt.savefig("/home/mant_hr/project_x/Thesis defence/LSTM_98.png")
plt.show()
