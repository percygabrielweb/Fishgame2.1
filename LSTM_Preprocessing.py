import pandas as pd

# Load the data from CSV
csv_file = 'tracking_data_pellets_only.csv'  # Replace with your actual file path
data = pd.read_csv(csv_file)

# Group by 'track_id' and count the number of frames for each track
sequence_lengths = data.groupby('track_id').size()

# Calculate descriptive statistics
statistics = sequence_lengths.describe()

# Specifically, we're interested in mean, 50% (median), and 75% (Q3)
mean_length = statistics['mean']
median_length = statistics['50%']
q3_length = statistics['75%']

print(f'Mean sequence length: {mean_length}')
print(f'Median (50%) sequence length: {median_length}')
print(f'Q3 (75%) sequence length: {q3_length}')
