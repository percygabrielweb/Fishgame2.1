import pandas as pd
import re

# Function to parse the position strings
def extract_positions(position_str):
    # This function extracts all (x, y) pairs from the position string
    # and converts them to a list of tuples (float, float)
    position_tuples = re.findall(r'\(([^,]+),([^)]+)\)', position_str)
    return [(float(x), float(y)) for x, y in position_tuples]

# Load the data from CSV
csv_file = 'tracking_data_labeled.csv'  # Replace with your actual file path
data = pd.read_csv(csv_file)

# Initialize a dictionary to hold the processed data
processed_data = {}

# Iterate over each row in the DataFrame
for index, row in data.iterrows():
    pellet_id = row['id']
    label = row['label']
    position_str = row['positions']
    positions = extract_positions(position_str)
    # Extract only the 'x' positions if needed
    x_positions = [pos[0] for pos in positions]
    # Store in the dictionary
    processed_data[pellet_id] = {'label': label, 'x_positions': x_positions}

# Now, processed_data contains the 'x' positions for each pellet

import numpy as np
import matplotlib.pyplot as plt

# Assuming 'processed_data' is your data structure from the previous steps
# and contains the 'x' positions for each pellet
x_positions = processed_data[3]['x_positions']

# Number of sample points
N = len(x_positions)
# Sample spacing
T = 1.0  # Change this if your frames represent different time intervals
x = np.linspace(0.0, N*T, N, endpoint=False)

# Perform the FFT
yf = np.fft.fft(x_positions)
xf = np.fft.fftfreq(N, T)[:N//2]

# Plotting the results
plt.figure()
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.title('FFT of Pellet Movements (Pellet ID: 4)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()
