import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch_geometric.transforms import LaplacianLambdaMax
from torch.nn import Parameter
from torch_geometric.nn import ChebConv
from torch_geometric.nn.inits import glorot, zeros
from math import radians, sin, cos, sqrt, atan2

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
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def forward(self, X, edge_index, edge_weight=None, H=None, C=None, lambda_max=None):
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        
        I = torch.matmul(X, self.W_i) + self.conv_i(H, edge_index, edge_weight, lambda_max=lambda_max) + self.b_i
        I = torch.sigmoid(I)
        
        F = torch.matmul(X, self.W_f) + self.conv_f(H, edge_index, edge_weight, lambda_max=lambda_max) + self.b_f
        F = torch.sigmoid(F)
        
        T = torch.matmul(X, self.W_c) + self.conv_c(H, edge_index, edge_weight, lambda_max=lambda_max) + self.b_c
        T = torch.tanh(T)
        
        C = F * C + I * T
        
        O = torch.matmul(X, self.W_o) + self.conv_o(H, edge_index, edge_weight, lambda_max=lambda_max) + self.b_o
        O = torch.sigmoid(O)
        
        H = O * torch.tanh(C)
        
        return H, C


# Function to calculate Haversine distance between two geographical points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Load and preprocess dataset
file_path = '/home/mant_hr/project_x/Final presentation/EV_station_data.csv'
df = pd.read_csv(file_path)

# Filter the data for 10 EV stations (assuming `tower_id` represents the stations)
station_ids = df['tower_id'].unique()[:9]
df_ev_stations = df[df['tower_id'].isin(station_ids)]

# Pivot the data to have stations as columns and time steps as rows
df_pivot = df_ev_stations.pivot(index='timestamp', columns='tower_id', values='residual')
df_pivot.fillna(0, inplace=True)
# Normalize the `kwh` values for each station
scaler = MinMaxScaler()
kwh_normalized = scaler.fit_transform(df_pivot)

# Split the data into train, validation, and test sets (70% train, 15% validation, 15% test)
train_data, test_data = train_test_split(kwh_normalized, test_size=0.15, shuffle=False)
train_data, val_data = train_test_split(train_data, test_size=0.15 / 0.85, shuffle=False)

# Convert data to PyTorch tensors
train_tensor = torch.tensor(train_data, dtype=torch.float)
val_tensor = torch.tensor(val_data, dtype=torch.float)
test_tensor = torch.tensor(test_data, dtype=torch.float)

# Get unique stations with their latitude and longitude
station_coords = df_ev_stations[['tower_id', 'tower_latitude', 'tower_longitude']].drop_duplicates().reset_index(drop=True)

# Calculate pairwise distances between all stations
num_stations = len(station_coords)
kwh_data = df_pivot

# Calculate correlation matrix for kwh time-series data
kwh_correlation = kwh_data.corr()

# Define thresholds for distance and correlation
distance_threshold = 50  # in kilometers
correlation_threshold = 0.7  # correlation threshold

# Create an empty adjacency matrix
adj_matrix_combined = np.zeros((num_stations, num_stations))

# Fill the adjacency matrix based on distance and correlation thresholds
for i in range(num_stations):
    for j in range(i + 1, num_stations):
        distance = haversine(station_coords.loc[i, 'tower_latitude'], station_coords.loc[i, 'tower_longitude'],
                             station_coords.loc[j, 'tower_latitude'], station_coords.loc[j, 'tower_longitude'])
        correlation = kwh_correlation.iloc[i, j]
        
        # Check if both distance and correlation are within thresholds
        if distance <= distance_threshold and correlation >= correlation_threshold:
            adj_matrix_combined[i, j] = 1
            adj_matrix_combined[j, i] = 1  # The graph is undirected

# Convert adjacency matrix to sparse edge index format for PyTorch Geometric
edge_index, edge_weight = dense_to_sparse(torch.tensor(adj_matrix_combined, dtype=torch.float))

# Create PyTorch Geometric data objects for train, val, and test sets
train_data = Data(x=train_tensor, edge_index=edge_index, edge_attr=edge_weight)
val_data = Data(x=val_tensor, edge_index=edge_index, edge_attr=edge_weight)
test_data = Data(x=test_tensor, edge_index=edge_index, edge_attr=edge_weight)

# Define the GC-LSTM model
in_channels = train_tensor.shape[1]  # The number of stations
out_channels = 256  # Output channels (adjustable based on model needs)
K = 4  # Chebyshev filter size

# Initialize the GC-LSTM model
model = GCLSTM(in_channels=in_channels, out_channels=out_channels, K=K)

# Add a fully connected layer to map the output back to the input dimension (in_channels)
fc = torch.nn.Linear(out_channels, in_channels)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
fc = fc.to(device)
train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

# Lambda max (for graph Laplacian normalization)
lambda_max = LaplacianLambdaMax()(train_data).lambda_max

# Check if lambda_max is a tensor or a float
if isinstance(lambda_max, torch.Tensor):
    lambda_max = lambda_max.to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(list(model.parameters()) + list(fc.parameters()), lr=0.001)
criterion = torch.nn.MSELoss()

# Training loop with validation and loss tracking
epochs = 100
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Reset hidden and cell states at each forward pass
    H, C = None, None
    
    # Forward pass through the GC-LSTM model (training set)
    H, C = model(train_data.x, train_data.edge_index, edge_weight=train_data.edge_attr, H=H, C=C, lambda_max=lambda_max)
    
    # Pass the hidden state through the fully connected layer to match the input dimension
    train_output = fc(H)
    
    # Calculate training loss
    train_loss = criterion(train_output, train_data.x)
    train_loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        H_val, C_val = model(val_data.x, val_data.edge_index, edge_weight=val_data.edge_attr, H=None, C=None, lambda_max=lambda_max)
        val_output = fc(H_val)
        val_loss = criterion(val_output, val_data.x)
    
    # Store losses
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}')

# Save the loss curve
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Train vs Validation Loss")
plt.savefig("train_val_loss_curve.png")
plt.show()

# Test set prediction
model.eval()
with torch.no_grad():
    H_test, C_test = model(test_data.x, test_data.edge_index, edge_weight=test_data.edge_attr, H=None, C=None, lambda_max=lambda_max)
    test_output = fc(H_test)

    # Inverse transform the predictions and actual values back to original scale
    predictions = scaler.inverse_transform(test_output.cpu().numpy())
    actual = scaler.inverse_transform(test_data.x.cpu().numpy())

# Number of towers (stations)
num_towers = test_data.x.shape[1]

# Plot actual vs predicted for each tower
for tower_id in range(num_towers):
    plt.figure()
    plt.plot(actual[-1:, tower_id], label="Actual", color='b')
    plt.plot(predictions[-1:, tower_id], label="Predicted", color='r')
    plt.xlabel("Timesteps")
    plt.ylabel("kWh")
    plt.legend()
    plt.title(f"Actual vs Predicted kWh on Test Data for Tower {tower_id+1}")
    plt.savefig(f"actual_vs_predicted_tower_{tower_id+1}.png")
    plt.show()

