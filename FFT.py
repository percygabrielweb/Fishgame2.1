import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv("tracking_data_pellets_only.csv")

# Convert 'frame' and 'track_id' columns to integers
df['frame'] = df['frame'].astype(int)
df['track_id'] = df['track_id'].astype(int)

# Group the data by 'track_id'
grouped = df.groupby('track_id')

# Dictionary to store interpolated data for each track
interpolated_data = {}

# Iterate over each group (track)
for track_id, group_df in grouped:
    # Sort the group by 'frame' number
    group_df = group_df.sort_values(by='frame')
    
    # Interpolate missing values if any
    group_df = group_df.interpolate(method='linear', axis=0)
    
    # Store the interpolated data for this track
    interpolated_data[track_id] = group_df
print(df[df['track_id']==1].head(40))

# Now 'interpolated_data' contains the segmented and interpolated data for each track
# You can access the data for a specific track like this:
# interpolated_data[track_id]


# Function to perform FFT on x and y coordinates
def perform_fft(data):
    # Sampling rate (assuming the frames are evenly spaced)
    sampling_rate = 1  # you can adjust this based on your actual frame rate

    # Number of samples
    n = len(data)

    # Time vector (assuming frames as time intervals)
    time_vector = np.arange(0, n) / sampling_rate

    # FFT for x and y coordinates
    fft_x = np.fft.fft(data['x'])
    fft_y = np.fft.fft(data['y'])

    # Frequency vector
    freq = np.fft.fftfreq(n, d=1/sampling_rate)

    # Plot FFT results
    plt.figure(figsize=(10, 6))
    plt.plot(freq[:n//2], np.abs(fft_x)[:n//2], label='FFT x')
    plt.plot(freq[:n//2], np.abs(fft_y)[:n//2], label='FFT y')
    plt.title('FFT Analysis')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
for track_id, data in interpolated_data.items():
    perform_fft(data)