import cv2
import os
from natsort import natsorted

# Specify the path for your folder containing images
input_folder_path = "dataset/images"
output_video_path = "pygame_movie.mp4"
desired_video_length_in_seconds = 60  # Set the desired video length here

# Get all files from the folder
images = [img for img in os.listdir(input_folder_path) if img.endswith(".jpg")]

# Sort the images by name
images = natsorted(images)

# Read the first image to get the width and height
frame = cv2.imread(os.path.join(input_folder_path, images[0]))
height, width, layers = frame.shape

# Calculate the frame rate needed to achieve the desired video length
frame_rate = len(images) / desired_video_length_in_seconds

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

for image in images:
    img_path = os.path.join(input_folder_path, image)
    img = cv2.imread(img_path)
    video.write(img)

# Release the video writer
video.release()