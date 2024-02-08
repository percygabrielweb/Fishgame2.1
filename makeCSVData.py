from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best.pt')

# Open the video file
video_path = "pygame_movie.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize the track history with more detailed tracking information
track_history = defaultdict(lambda: {'x': [], 'y': [], 'frames': []})

# Define the class ID for pellets
pellet_class_id = 0  # Assuming '0' is the class ID for pellets

frame_count = 0  # Initialize frame count

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        
        if results[0].boxes:  # Check if there are any detected boxes
            # Get the boxes, track IDs, and class IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            # Update track history for pellets
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if class_id == pellet_class_id:
                    x, y, _, _ = box
                    track_history[track_id]['x'].append(x.item())
                    track_history[track_id]['y'].append(y.item())
                    track_history[track_id]['frames'].append(frame_count)

        frame_count += 1  # Increment frame count

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# Calculate summary statistics and write to a CSV
data = []
for track_id, track in track_history.items():
    if len(track['y']) > 1:  # Ensure there are enough points to calculate speed
        y_speed = np.diff(track['y']).mean()  # Average Y-speed
        x_variance = np.var(track['x'])  # Variance of X coordinates
        x_range = abs(max(track['x']) - min(track['x']))  # Range of X coordinates
        data.append([track_id, y_speed, x_variance, x_range])

# Create a DataFrame and save as CSV
df = pd.DataFrame(data, columns=['track_id', 'avg_y_speed', 'x_variance', 'x_range'])
df.to_csv('track_summary_statistics.csv', index=False)
