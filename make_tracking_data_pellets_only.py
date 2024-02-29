from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import csv


'''This code will make a csv files ''tracking_data_pellets_only.csv'' based on tracking done by YOLO on pygame_movie.mp4,'''

# Load the YOLOv8 model
model = YOLO('best.pt')

# Open the video file
video_path = "pygame_movie.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Define the class ID for pellets
pellet_class_id = 0  # Assuming '0' is the class ID for pellets

# Open a file to write the tracking data for pellets only
with open('tracking_data_pellets_only.csv', 'w') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame', 'track_id', 'x', 'y'])  # Header for CSV format
    
    frame_count = 0  # Initialize frame count
    
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)
            
            if results[0].boxes:  # Check if there are any detected boxes
                # Get the boxes, track IDs, and class IDs using the 'cls' attribute
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()  # Use 'cls' attribute for class IDs

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks and write to file if the object is a pellet
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    if class_id == pellet_class_id:  # Check if the object is classified as pellet
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append(float(x))# Append x directly
                        track.append(float(y))# Append y directly
                        csv_writer.writerow([frame_count, track_id, x.item(), y.item()])
                        
                        # Draw the tracking lines
                        if track:  # Check if there are points to plot
                            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=1)
            else:
                # If no boxes are detected, simply show the current frame
                annotated_frame = np.array(frame)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            frame_count += 1  # Increment frame count

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Break the loop if the end of the video is reached
            break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
