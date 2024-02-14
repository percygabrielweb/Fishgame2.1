import pandas as pd

# Assuming 'your_file.csv' is the path to your CSV file
csv_file_path = 'tracking_data_labeled.csv'#'track_summary_statistics.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Display the first few rows of the DataFrame
print(df.head(20))

import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('tracking_data_labeled.csv')

# Preprocessing to convert positions from string to list of tuples
def parse_positions(positions_str):
    positions_str = positions_str.strip('"')
    positions_list = positions_str.split(';')
    return [tuple(map(float, pos.strip('()').split(','))) for pos in positions_list]

df['positions'] = df['positions'].apply(parse_positions)

# Calculate features
features = []

for _, row in df.iterrows():
    positions = row['positions']
    y_speeds = [positions[i+1][1] - positions[i][1] for i in range(len(positions)-1)]
    x_positions = [pos[0] for pos in positions]
    
    avg_y_speed = np.mean(y_speeds) if y_speeds else 0
    x_variance = np.var(x_positions) if x_positions else 0
    x_range = max(x_positions) - min(x_positions) if x_positions else 0
    
    features.append([row['id'], row['label'], avg_y_speed, x_variance, x_range])

# Create a DataFrame for the features
features_df = pd.DataFrame(features, columns=['id', 'label', 'avg_y_speed', 'x_variance', 'x_range'])

# Save to CSV
features_df.to_csv('feature_data.csv', index=False)

df2 = pd.read_csv('feature_data.csv')
print(df2.head())