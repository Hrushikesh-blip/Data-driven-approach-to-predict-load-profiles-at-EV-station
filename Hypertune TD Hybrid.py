import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch_geometric.transforms import LaplacianLambdaMax
from torch.nn import Parameter
from torch_geometric.nn import ChebConv
from torch_geometric.nn.inits import glorot, zeros
import matplotlib.pyplot as plt
import optuna

# Ensure necessary variables are globally available
global train_tensor_x, train_tensor_y, val_tensor_x, val_tensor_y, train_data, val_data, test_data
global edge_index, edge_weight, num_towers, lambda_max, scaler_kwh

# Function to create sequences from time series data
def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        target = data[i + sequence_length]
        sequences.append(sequence)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

# Define the GCLSTM model class
class GCLSTM(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K, normalization="sym", bias=True):
        super(GCLSTM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _create_input_gate_parameters_and_layers(self):
        self.conv_i = ChebConv(self.out_channels, self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.W_i = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):
        self.conv_f = ChebConv(self.out_channels, self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.W_f = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):
        self.conv_c = ChebConv(self.out_channels, self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.W_c = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):
        self.conv_o = ChebConv(self.out_channels, self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.W_o = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _set_parameters(self):
        glorot(self.W_i)
        glorot(self.W_f)
        glorot(self.W_c)
        glorot(self.W_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels, device=X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels, device=X.device)
        return C

    def forward(self, X, edge_index, edge_weight=None, H=None, C=None, lambda_max=None):
        batch_size, sequence_length, num_towers, num_features = X.shape
        X = X.view(batch_size * num_towers, sequence_length, num_features)

        H = self._set_hidden_state(X[:, 0, :], H)
        C = self._set_cell_state(X[:, 0, :], C)

        for t in range(sequence_length):
            X_t = X[:, t, :]

            I = torch.matmul(X_t, self.W_i) + self.conv_i(H, edge_index, edge_weight, lambda_max=lambda_max) + self.b_i
            I = torch.sigmoid(I)

            F = torch.matmul(X_t, self.W_f) + self.conv_f(H, edge_index, edge_weight, lambda_max=lambda_max) + self.b_f
            F = torch.sigmoid(F)

            T = torch.matmul(X_t, self.W_c) + self.conv_c(H, edge_index, edge_weight, lambda_max=lambda_max) + self.b_c
            T = torch.tanh(T)

            C = F * C + I * T

            O = torch.matmul(X_t, self.W_o) + self.conv_o(H, edge_index, edge_weight, lambda_max=lambda_max) + self.b_o
            O = torch.sigmoid(O)

            H = O * torch.tanh(C)
            
        return H, C

# Load and preprocess dataset
file_path = '/home/mant_hr/project_x/complete_dataset_with_components_and_coords.csv'
df = pd.read_csv(file_path)

# Filter for the first 10 EV stations
station_ids = df['tower_id'].unique()[:10]
df_ev_stations = df[df['tower_id'].isin(station_ids)]

# Extract relevant features (kWh, daily_seasonality, weekly_seasonality, residuals)
df_features = df_ev_stations[['timestamp', 'tower_id', 'kwh', 'daily_seasonality', 'weekly_seasonality', 'residuals']]

# Pivot the data to have stations as columns and time steps as rows for each feature
df_pivot_kwh = df_features.pivot(index='timestamp', columns='tower_id', values='kwh')
df_pivot_daily = df_features.pivot(index='timestamp', columns='tower_id', values='daily_seasonality')
df_pivot_weekly = df_features.pivot(index='timestamp', columns='tower_id', values='weekly_seasonality')
df_pivot_residuals = df_features.pivot(index='timestamp', columns='tower_id', values='residuals')

# Normalize the feature columns
scaler_kwh = MinMaxScaler()
scaler_daily = MinMaxScaler()
scaler_weekly = MinMaxScaler()
scaler_residuals = MinMaxScaler()

kwh_normalized = scaler_kwh.fit_transform(df_pivot_kwh)
daily_normalized = scaler_daily.fit_transform(df_pivot_daily)
weekly_normalized = scaler_weekly.fit_transform(df_pivot_weekly)
residuals_normalized = scaler_residuals.fit_transform(df_pivot_residuals)

# Concatenate the normalized features
features_combined = np.stack([kwh_normalized, daily_normalized, weekly_normalized, residuals_normalized], axis=-1)

# Create sequences
sequence_length = 24
X_train, y_train = create_sequences(features_combined[:int(0.7 * len(features_combined))], sequence_length)
X_val, y_val = create_sequences(features_combined[int(0.7 * len(features_combined)):int(0.85 * len(features_combined))], sequence_length)
X_test, y_test = create_sequences(features_combined[int(0.85 * len(features_combined)):], sequence_length)

# Only use kWh (1st index) as the target
y_train = y_train[:, :, 0]
y_val = y_val[:, :, 0]
y_test = y_test[:, :, 0]

# Convert sequences to PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_tensor_x = torch.tensor(X_train, dtype=torch.float).to(device)
train_tensor_y = torch.tensor(y_train, dtype=torch.float).to(device)
val_tensor_x = torch.tensor(X_val, dtype=torch.float).to(device)
val_tensor_y = torch.tensor(y_val, dtype=torch.float).to(device)
test_tensor_x = torch.tensor(X_test, dtype=torch.float).to(device)
test_tensor_y = torch.tensor(y_test, dtype=torch.float).to(device)

# Create adjacency matrix based on distance and correlation
station_coords = df_ev_stations[['tower_id', 'tower_latitude', 'tower_longitude']].drop_duplicates().reset_index(drop=True)

# Assuming adjacency matrix creation using predefined threshold for distance and correlation
adj_matrix_combined = np.zeros((len(station_ids), len(station_ids)))

distance_threshold = 50
correlation_threshold = 0.7
kwh_correlation = df_pivot_kwh.corr()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

num_towers = 10

for i in range(len(station_ids)):
    for j in range(i + 1, len(station_ids)):
        distance = haversine(station_coords.loc[i, 'tower_latitude'], station_coords.loc[i, 'tower_longitude'],
                             station_coords.loc[j, 'tower_latitude'], station_coords.loc[j, 'tower_longitude'])
        correlation = kwh_correlation.iloc[i, j]
        
        if distance <= distance_threshold and correlation >= correlation_threshold:
            adj_matrix_combined[i, j] = 1
            adj_matrix_combined[j, i] = 1

# Convert adjacency matrix to sparse edge index format for PyTorch Geometric
edge_index, edge_weight = dense_to_sparse(torch.tensor(adj_matrix_combined, dtype=torch.float).to(device))

# Create PyTorch Geometric data objects for train, val, and test sets
train_data = Data(x=train_tensor_x, edge_index=edge_index, edge_attr=edge_weight)
val_data = Data(x=val_tensor_x, edge_index=edge_index, edge_attr=edge_weight)
test_data = Data(x=test_tensor_x, edge_index=edge_index, edge_attr=edge_weight)
criterion = torch.nn.MSELoss()
# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to optimize
    out_channels = trial.suggest_int('out_channels', 16, 128)  # Tune output channels
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)            # Tune learning rate
    K = trial.suggest_int('K', 2, 6)                           # Tune Chebyshev filter size

    # Redefine the model and optimizer with trial parameters
    model = GCLSTM(in_channels=train_tensor_x.shape[3], out_channels=out_channels, K=K).to(device)
    fc = torch.nn.Linear(out_channels, 1).to(device)
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(fc.parameters()), lr=lr)
    criterion = torch.nn.MSELoss()

    train_losses = []
    val_losses = []

    epochs = 10  # You can increase this for more thorough tuning

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        H, C = model(train_data.x, train_data.edge_index, edge_weight=train_data.edge_attr)
        train_output = fc(H).view(-1, num_towers)
        train_tensor_y_flat = train_tensor_y.view(-1, num_towers)

        train_loss = criterion(train_output, train_tensor_y_flat)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            H_val, C_val = model(val_data.x, val_data.edge_index, edge_weight=val_data.edge_attr)
            val_output = fc(H_val).view(-1, num_towers)
            val_loss = criterion(val_output, val_tensor_y.view(-1, num_towers))

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

    # Return validation loss for Optuna to minimize
    return val_loss.item()

# Create the Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)  # Adjust n_trials based on computation resources

# Get the best parameters found by Optuna
best_params = study.best_params
print(f"Best parameters: {best_params}")

# Train and evaluate the final model with the best hyperparameters
best_out_channels = best_params['out_channels']
best_lr = best_params['lr']
best_K = best_params['K']

# Redefine and train the model using the best hyperparameters
model = GCLSTM(in_channels=train_tensor_x.shape[3], out_channels=best_out_channels, K=best_K).to(device)
fc = torch.nn.Linear(best_out_channels, 1).to(device)
optimizer = torch.optim.Adam(list(model.parameters()) + list(fc.parameters()), lr=best_lr)

# Training loop with the best hyperparameters
train_losses = []
val_losses = []
epochs = 100  # You can adjust this based on your computation budget

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    H, C = model(train_data.x, train_data.edge_index, edge_weight=train_data.edge_attr)
    train_output = fc(H).view(-1, num_towers)
    train_tensor_y_flat = train_tensor_y.view(-1, num_towers)

    train_loss = criterion(train_output, train_tensor_y_flat)
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        H_val, C_val = model(val_data.x, val_data.edge_index, edge_weight=val_data.edge_attr)
        val_output = fc(H_val).view(-1, num_towers)
        val_loss = criterion(val_output, val_tensor_y.view(-1, num_towers))

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}')

# Plot the training and validation loss curves
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Train vs Validation Loss")
plt.savefig("train_val_loss_curve.png")
plt.show()

# Test set evaluation
model.eval()
with torch.no_grad():
    H_test, C_test = model(test_data.x, test_data.edge_index, edge_weight=test_data.edge_attr)
    test_output = fc(H_test).view(-1, num_towers)

    predictions = scaler_kwh.inverse_transform(test_output.cpu().numpy())
    actual = scaler_kwh.inverse_transform(test_tensor_y.cpu().numpy())

mse = mean_squared_error(actual, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actual, predictions)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R² Score: {r2}')

# Plot actual vs predicted kWh for each tower
for tower_id in range(predictions.shape[1]):
    plt.figure(figsize=(10, 6))
    plt.plot(actual[:, tower_id], label="Actual", color='b')
    plt.plot(predictions[:, tower_id], label="Predicted", color='r')
    plt.xlabel("Time Steps")
    plt.ylabel("kWh")
    plt.title(f"Actual vs Predicted kWh for Tower {tower_id + 1}")
    plt.legend()
    plt.savefig(f"actual_vs_predicted_tower_{tower_id+1}.png")
    plt.show()
