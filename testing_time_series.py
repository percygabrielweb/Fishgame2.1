import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("time_series.csv")

# Define IDs to plot
ids = [1, 2, 3]

# Filter data for given IDs
filtered_df = df[df["  track_id  "].isin(ids)]


# Create separate plots for each ID
for track_id in ids:
    track_data = filtered_df[filtered_df["track_id"] == track_id]
    plt.scatter(track_data["x"], track_data["y"], label=f"ID {track_id}")

# Customize plot
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Movement of IDs 1, 2, and 3")
plt.legend()

# Show the plot
plt.show()
