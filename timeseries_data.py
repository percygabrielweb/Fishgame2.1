import pandas as pd

# Define function to parse a single row
def parse_row(row):
    frame = row["frame"]
    track_id = row["track_id"]

    # Extract numerical part from "tensor(value)" strings
    x = float(row["x"].split("(")[1].split(")")[0])
    y = float(row["y"].split("(")[1].split(")")[0])

    timestamp = frame / 30

    return pd.DataFrame({
        "track_id": [track_id],
        "timestamp": [timestamp],
        "x": [x],
        "y": [y]
    })


# Read CSV data
data = pd.read_csv("tracking_data_pellets_only.csv")

# Create separate DataFrames for each track
track_dfs = []
for track_id in data["track_id"].unique():
    track_data = data[data["track_id"] == track_id]
    track_df = track_data.apply(parse_row, axis=1).reset_index(drop=True)
    track_dfs.append(track_df)

# Concatenate DataFrames for all tracks
time_series_df = pd.concat(track_dfs, ignore_index=True)

# Print the first few rows to verify
print(time_series_df.head())

# Save the time series DataFrame
time_series_df.to_csv("time_series.csv", index=False)
