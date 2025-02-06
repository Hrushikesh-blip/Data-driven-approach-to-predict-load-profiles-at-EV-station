import pandas as pd

def parse_date(date_str):
    formats = ['%Y-%m-%d %H:%M:%S%z', '%Y-%m-%d %H:%M:%S.%f%z', '%Y-%m-%d %H:%M:%S.%f']
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            pass
    # If parsing fails for all formats, print the timestamp and return NaT (Not a Time)
    print(f"Error parsing timestamp: {date_str}")
    return pd.NaT

file_path = '/home/mant_hr/project_x/Data preprocessing/ev_stations_data_with_diff.csv'  
data = pd.read_csv(file_path)

tower_ids = [98, 137, 175, 213, 214, 215, 216, 217, 218]
tower_98_data = data[data['tower_id'] == 98]
output_path = '/home/mant_hr/project_x/Data preprocessing/tower98.csv'
tower_98_data.to_csv(output_path, index=False)
# Ensure kWh_diff is non-negative
data['kWh_diff'] = data['kWh_diff'].apply(lambda x: max(x, 0))

# Set time_diff to zero where kWh_diff is zero
data.loc[data['kWh_diff'] == 0, 'time_diff'] = 0

# Calculate the power column
data['power'] = data.apply(
    lambda row: row['kWh_diff'] / (row['time_diff'] / 3600) if row['time_diff'] > 0 else 0, axis=1
)

data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')  # Handle parsing errors with 'coerce'

# Check for rows with invalid timestamps and drop them
if data['timestamp'].isna().any():
    print("Warning: Invalid timestamps found. These rows will be dropped.")
    data = data.dropna(subset=['timestamp'])

# Set the timestamp column as the index
data.set_index('timestamp', inplace=True)

# Verify that the index is a DatetimeIndex
if not isinstance(data.index, pd.DatetimeIndex):
    raise TypeError("The index is not a DatetimeIndex. Ensure timestamp conversion is correct.")

# Resample power to a 1-hour resolution grouped by tower_id
aggregated_data = (
    data.groupby('tower_id')
    .resample('1H')['power']
    .sum()
    .reset_index()
)

# Save the aggregated dataset to a new CSV file
output_path = '/home/mant_hr/project_x/Data preprocessing/ev_stations_power_aggregated_hourly.csv'
aggregated_data.to_csv(output_path, index=False)

print(f"Aggregated data saved to: {output_path}")