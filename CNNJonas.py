import pandas as pd

csv_file = 'tracking_data_labeled.csv'  # Replace with your actual file path
data = pd.read_csv(csv_file)


import re

def extract_positions(position_str):
    # This function extracts all (x, y) pairs from the position string
    # and converts them to a list of x-coordinates (as floats).
    position_tuples = re.findall(r'\(([^,]+),([^)]+)\)', position_str)
    return [float(x) for x, y in position_tuples]

processed_data = {}

for index, row in data.iterrows():
    pellet_id = row['id']
    label = row['label']
    position_str = row['positions']
    x_positions = extract_positions(position_str)
    processed_data[pellet_id] = {'label': label, 'x_positions': x_positions}


from keras.preprocessing.sequence import pad_sequences

def adjust_sequences(data, sequence_length=87):
    # Extract x_positions from each sequence and store them in a list
    x_sequences = [item['x_positions'] for item in data.values()]
    
    # Pad or truncate the sequences
    x_adjusted = pad_sequences(x_sequences, maxlen=sequence_length, dtype='float32', padding='post', truncating='post')
    
    # Update the original data with adjusted sequences
    updated_data = {}
    for idx, key in enumerate(data.keys()):
        updated_data[key] = {
            'label': data[key]['label'],
            'x_positions': x_adjusted[idx].tolist()  # Convert from numpy array back to list
        }
    return updated_data

adjusted_data = adjust_sequences(processed_data)


import numpy as np

def apply_fft(data):
    for key, item in data.items():
        x_positions = item['x_positions']
        yf = np.fft.fft(x_positions)
        # Keep only the positive frequencies and normalize
        magnitude = np.abs(yf)[:len(yf)//2]
        data[key]['fft'] = (magnitude / np.max(magnitude)).tolist()
    return data

fft_data = apply_fft(adjusted_data)

def prepare_for_cnn(data):
    all_fft = np.array([item['fft'] for item in data.values()])
    labels = np.array([item['label'] for item in data.values()])
    return all_fft, labels

X, y = prepare_for_cnn(fft_data)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)