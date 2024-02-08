import pandas as pd

# Assuming 'your_file.csv' is the path to your CSV file
csv_file_path = 'track_summary_statistics.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Display the first few rows of the DataFrame
print(df.head(20))