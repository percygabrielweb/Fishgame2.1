import pandas as pd
from joblib import load  # Depending on your sklearn version, you might need to adjust this import

# Step 1: Load your data
data_path = "track_summary_statistics.csv"  # Update this path to your CSV file
data = pd.read_csv(data_path)

# Assuming your DataFrame's relevant features are named 'avg_y_speed', 'x_variance', and 'x_range'
X_new = data[['avg_y_speed', 'x_variance', 'x_range']]

# Step 3: Load the saved model
model = load("random_forest_classifier.joblib")  # Update the filename to your saved model

# Step 4: Make predictions
predictions = model.predict(X_new)

# You can now add these predictions to your DataFrame or analyze them as needed.
data['predicted_label'] = predictions

# Optionally, save the DataFrame with predictions back to a new CSV
data.to_csv("labeled_data_predictions____.csv", index=False)
