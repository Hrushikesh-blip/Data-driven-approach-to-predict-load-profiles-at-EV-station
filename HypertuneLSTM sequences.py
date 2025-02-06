import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import torch.nn.functional as F
import optuna
import optuna.visualization as vis
# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, prediction_length):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_length = prediction_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * prediction_length)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = out.view(out.size(0), self.prediction_length, -1)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

file_path = '/home/mant_hr/project_x/decomposed_ev_station_kwh_by_tower.csv'
df = pd.read_csv(file_path)
station_ids = df['tower_id'].unique()
df_ev_stations = df[df['tower_id'].isin(station_ids)]
df_pivot = df_ev_stations.pivot(index='timestamp', columns='tower_id', values=['kwh'])

scalers = {}
for feature in df_pivot.columns.levels[0]:
    feature_df = df_pivot[feature]
    scaler = MinMaxScaler()
    df_pivot[feature] = pd.DataFrame(scaler.fit_transform(feature_df), columns=feature_df.columns, index=feature_df.index)
    scalers[feature] = scaler

sequence_length = 24
prediction_length = 24

def create_sequences(data, sequence_length, target_col_idx, num_towers, lead_time):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length - prediction_length - lead_time):
        sequence = data[i:i + sequence_length]
        target = data[i + sequence_length:i + sequence_length + prediction_length, target_col_idx:target_col_idx + num_towers]
        sequences.append(sequence)
        targets.append(target)
    sequences = np.array(sequences)
    targets = np.array(targets)
    return sequences, targets

lead_time = 0
X, y = create_sequences(df_pivot.values, sequence_length, target_col_idx=0, num_towers=10, lead_time=lead_time)

train_size = int(len(X) * 0.64)
val_size = int(len(X) * 0.16)
test_size = len(X) - train_size - val_size

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

train_tensor_x = torch.tensor(X_train, dtype=torch.float32).to(device)
train_tensor_y = torch.tensor(y_train, dtype=torch.float32).to(device)
val_tensor_x = torch.tensor(X_val, dtype=torch.float32).to(device)
val_tensor_y = torch.tensor(y_val, dtype=torch.float32).to(device)
test_tensor_x = torch.tensor(X_test, dtype=torch.float32).to(device)
test_tensor_y = torch.tensor(y_test, dtype=torch.float32).to(device)

def objective(trial):
    input_size = 10
    hidden_size = trial.suggest_int('hidden_size', 32, 128)
    output_size = 10
    num_layers = trial.suggest_int('num_layers', 2, 4)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    criterion_type = trial.suggest_categorical('criterion', ['mse', 'smooth_l1'])

    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, prediction_length=prediction_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() if criterion_type == 'mse' else nn.SmoothL1Loss()

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    epochs = 50
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_tensor_x)
        train_loss = criterion(outputs, train_tensor_y)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tensor_x)
            val_loss = criterion(val_outputs, val_tensor_y)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break
    

    train_outputs = model(train_tensor_x).cpu().detach().numpy()
    train_targets = train_tensor_y.cpu().numpy()
    val_outputs = val_outputs.cpu().detach().numpy()
    val_targets = val_tensor_y.cpu().numpy()

    train_rmse = sqrt(mean_squared_error(train_targets.flatten(), train_outputs.flatten()))
    train_mae = mean_absolute_error(train_targets.flatten(), train_outputs.flatten())
    val_rmse = sqrt(mean_squared_error(val_targets.flatten(), val_outputs.flatten()))
    val_mae = mean_absolute_error(val_targets.flatten(), val_outputs.flatten())

    trial.set_user_attr("train_rmse", train_rmse)
    trial.set_user_attr("train_mae", train_mae)
    trial.set_user_attr("val_rmse", val_rmse)
    trial.set_user_attr("val_mae", val_mae)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss(MSE)')
    plt.title('Training vs. Validation Loss')
    plt.legend()
    plt.show()
    return best_val_loss.item()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
optimization_history = vis.plot_optimization_history(study)
parallel_plot = vis.plot_parallel_coordinate(study)
parallel_plot.write_html("parallelplot.html")
# Display the plot
optimization_history.write_html("optimization_history.html")
# Print the top 5 trials with their parameters and metrics
trials = sorted(study.trials, key=lambda x: x.value)[:5]
print("\nTop 5 Trials:")
for i, trial in enumerate(trials, 1):
    print(f"\nTrial {i}:")
    print(f"  Parameters: {trial.params}")
    print(f"  Training RMSE: {trial.user_attrs['train_rmse']:.4f}")
    print(f"  Training MAE: {trial.user_attrs['train_mae']:.4f}")
    print(f"  Validation RMSE: {trial.user_attrs['val_rmse']:.4f}")
    print(f"  Validation MAE: {trial.user_attrs['val_mae']:.4f}")
