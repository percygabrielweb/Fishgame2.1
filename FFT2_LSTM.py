import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
import numpy as np

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

# Print the raw output of the FFT
#print(yf)

# Assuming 'processed_data' is structured as {id: {'label': label, 'x_positions': [x1, x2, ...]}}
#for pellet_id, info in processed_data.items():
#    x_positions = info['x_positions']
#    N = len(x_positions)  # Number of sample points
#    T = 1.0  # Sample spacing, change this according to your actual time intervals
#
#    # Perform the FFT
#    yf = np.fft.fft(x_positions)
#    
#    # Print the FFT output
#    print(f"FFT output for Pellet ID {pellet_id}:")
#    print(yf)
#    print("\n")  # Adds a newline for better readability between different pellets' outputs

def adjust_sequences(data, sequence_length=87):
    # Extract x_positions from each sequence and store them in a list
    x_sequences = [item['x_positions'] for item in data.values()]

    # Pad or truncate the sequences
    # Note: We set padding='post' and truncating='post' to add/remove values at the end of the sequences
    x_adjusted = pad_sequences(x_sequences, maxlen=sequence_length, dtype='float32', padding='post', truncating='post')

    # Update the original data with adjusted sequences
    updated_data = {}
    for idx, key in enumerate(data.keys()):
        updated_data[key] = {
            'label': data[key]['label'],
            'x_positions': x_adjusted[idx].tolist()  # Convert from numpy array back to list
        }
    
    return updated_data

# Now, apply this function to your processed_data
adjusted_data = adjust_sequences(processed_data)





from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Assuming 'adjusted_data' is your dataset after padding/truncation
# Extract all x_positions in a list
all_x_positions = [item['x_positions'] for item in adjusted_data.values()]

# Convert list of lists into a 2D numpy array
all_x_positions_array = np.array(all_x_positions)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler to your data
# Note: MinMaxScaler expects a 2D array, so if your data is 1D, you might need to reshape it
# For our purposes, we're assuming it's already properly formatted as 2D
normalized_x_positions = scaler.fit_transform(all_x_positions_array.reshape(-1, 1))

# Reshape the normalized data back to its original shape
normalized_x_positions = normalized_x_positions.reshape(all_x_positions_array.shape)

# Update your dataset with normalized sequences
for i, key in enumerate(adjusted_data.keys()):
    adjusted_data[key]['x_positions'] = normalized_x_positions[i].tolist()


# Assuming 'normalized_x_positions' contains your normalized sequences
X = normalized_x_positions  # This is your features (input)

# And assuming you have your labels stored similarly
y = np.array([item['label'] for item in adjusted_data.values()])  # This is your target output


from sklearn.model_selection import train_test_split

# Split the data - 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Reshape the data for LSTM model
# 'time steps' is your sequence length, 'features' is 1 because you have only 'x_positions'
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))



from keras.models import Sequential
from keras.layers import LSTM, Dense

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], 1)))  # Adjust the number '50' based on your dataset and model complexity
model.add(Dense(1, activation='sigmoid'))  # Use 'sigmoid' for binary classification; for multi-class, use 'softmax' and adjust the number of units to match the number of classes

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Use 'categorical_crossentropy' for multi-class


--_____-this is where i am----, just wrote the code above as u can see...