import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
import wandb
wandb.init(project='your_project_name', entity='your_wandb_username')
# Function to extract x and y positions from the position string
def extract_positions(position_str):
    # This function extracts all (x, y) pairs from the position string
    # and converts them to a list of tuples (float, float)
    position_tuples = re.findall(r'\(([^,]+),([^)]+)\)', position_str)
    return [(float(x), float(y)) for x, y in position_tuples]

# Load the data from CSV
csv_file = 'tracking_data_labeled.csv'  # Replace with your actual file path
data = pd.read_csv(csv_file)

# Extract x and y positions
data['positions'] = data['positions'].apply(extract_positions)
data['x_positions'] = data['positions'].apply(lambda pos: [p[0] for p in pos])
data['y_positions'] = data['positions'].apply(lambda pos: [p[1] for p in pos])


# Flatten for normalization
all_x_positions = np.concatenate(data['x_positions']).reshape(-1, 1)
all_y_positions = np.concatenate(data['y_positions']).reshape(-1, 1)



# Initialize and apply MinMaxScaler
scaler = MinMaxScaler()
scaled_x_positions = scaler.fit_transform(all_x_positions)
scaled_y_positions = scaler.fit_transform(all_y_positions)

from keras.preprocessing.sequence import pad_sequences

SEQUENCE_LENGTH = 87  # Set this to the desired sequence length

# Pad or truncate x_positions and y_positions
data['x_positions'] = list(pad_sequences(data['x_positions'], maxlen=SEQUENCE_LENGTH, padding='post', truncating='post', dtype='float'))
data['y_positions'] = list(pad_sequences(data['y_positions'], maxlen=SEQUENCE_LENGTH, padding='post', truncating='post', dtype='float'))


# Combine x and y positions into a single feature set
combined_features = np.array([np.column_stack((x, y)) for x, y in zip(data['x_positions'], data['y_positions'])])

# Only x_positions for LSTM '''remove this if it is not needed'''
x_features = np.array([np.array(x) for x in data['x_positions']])
x_features = x_features.reshape((x_features.shape[0], SEQUENCE_LENGTH, 1))  # Reshape for LSTM


# Prepare labels for training
labels = np.array(data['label'])

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)
# Split the x_features and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_features, labels, test_size=0.2, random_state=42)


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Define the LSTM model
model = Sequential()
#change input shape to 87, 2 if we use the combined_features instead of 87, 1 for only x or only y
model.add(LSTM(64, input_shape=(87, 1), return_sequences=True))  # Adjust the number of LSTM units based on your data
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # Use 'softmax' if you have more than two classes

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary to check the architecture
model.summary()


# Fit the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

