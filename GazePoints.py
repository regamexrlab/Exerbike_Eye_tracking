import cv2
import numpy as np
from tkinter import filedialog
import tkinter as tk
from tqdm import tqdm  # Import tqdm
from moviepy.editor import VideoFileClip

# Define a temporary video file path
temp_path = "/Users/kaviprakashramalingam/Desktop/temp.mp4"     #Replace with local path
# Function to get user input for file paths and minutes to skip
def get_user_input():
    root = tk.Tk()
    root.withdraw()

    # 1. Get the video file name
    video_file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4")])

    # 2. Get the gaze data file
    gaze_data_path = filedialog.askopenfilename(title="Select Gaze Data File", filetypes=[("CSV Files", "*.csv")])

    # 3. Get the minutes to skip
    minutes_to_skip = float(input("Enter the minutes to skip: "))
    # minutes_to_skip = 1.06

    # 4. Get the end time to stop processing
    end_time = float(input("Enter the end time (in minutes) to stop processing: "))

    # 4. Get the destination path for the processed video file
    destination_path = filedialog.asksaveasfilename(title="Select Destination Path and File Name", filetypes=[("Video Files", "*.mp4")])


    root.destroy()
    return video_file_path, gaze_data_path, minutes_to_skip , end_time, destination_path

# Get user input
video_file, gaze_data_file, minutes_to_skip, end_time, destination_path = get_user_input()

# Load the video
video_capture = cv2.VideoCapture(video_file)

# Load gaze data from a CSV file (adjust the code to match your data format)
gaze_data = np.loadtxt(gaze_data_file, delimiter=',', skiprows=1)

# Round off timestamps in gaze_data to two decimal places
gaze_data[:, 0] = np.round(gaze_data[:, 0], 2)

# Get the frame rate of the video
fps = video_capture.get(cv2.CAP_PROP_FPS)

# Get the frame width and height
frame_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Define the circle radius
circle_radius = 10

# Define the coordinates of the building ROI
building_roi = {'x': 700, 'y': 400, 'width': 500, 'height': 500}

# Calculate the number of frames to skip (2 minutes)
# frames_to_skip = 1.05 * 60 * fps
if minutes_to_skip < 1:
    minutes_to_skip = ( minutes_to_skip / 60 ) * 100
frames_to_skip = int(minutes_to_skip * 60 * fps)
frames_skipped = 0

# Calculate the number of frames to process (end time)
frames_to_process = int(end_time * 60 * fps)
frames_processed = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(temp_path, fourcc, fps, (int(frame_width), int(frame_height)))

# Flag to check the first iteration
first_iteration = True

# Variables to track gaze duration on building ROI
total_gaze_duration = 0
in_building_roi = False
building_roi_enter_time = 0

# while video_capture.isOpened() and frames_processed < frames_to_process:   # and frame_count < 3600:
# Use tqdm for the progress bar
for _ in tqdm(range(frames_to_process), desc="Processing Frames", unit="frame"):
    ret, frame = video_capture.read()

    if not ret:
        break

    # Skip the initial frames
    if frames_skipped < frames_to_skip:
        frames_skipped += 1
        continue

    # Get the current timestamp based on the frame number and video frame rate
    current_timestamp = round(video_capture.get(cv2.CAP_PROP_POS_FRAMES) / fps, 2)

    if first_iteration:
        initial_timestamp = current_timestamp
        first_iteration = False

    current_timestamp = round(current_timestamp - initial_timestamp,2)
    # Find the corresponding row in the gaze data for the current timestamp
    matching_row = np.where(gaze_data[:, 0] == current_timestamp)[0]

     # If a matching row is found, overlay gaze points on the frame
    if matching_row.size > 0:

        # Convert gaze points from percentages to pixel values
        x_left = gaze_data[matching_row, 10].squeeze()
        y_left = gaze_data[matching_row, 11].squeeze()
        x_right = gaze_data[matching_row, 21].squeeze()
        y_right = gaze_data[matching_row, 22].squeeze()
        # Convert percentage values to pixel values based on the frame size
        x_left_pixel = ( x_left/100 ) * frame_width
        y_left_pixel = ( y_left/100 ) * frame_height
        x_right_pixel = ( x_right/100 ) * frame_width
        y_right_pixel = ( y_right/100 ) * frame_height
        
        # # Update building ROI coordinates based on the dynamic movement
        # building_roi['x'] = updated_x  # Replace with your logic to update x coordinate
        # building_roi['y'] = updated_y  # Replace with your logic to update y coordinate

# Check if the gaze points are inside the building ROI
        # if 10 <= current_timestamp <= 20:
        #     if (
        #         building_roi['x'] < x_left < building_roi['x'] + building_roi['width'] and
        #         building_roi['y'] < y_left < building_roi['y'] + building_roi['height']
        #     ):
        #         if not in_building_roi:
        #             # Enter building ROI
        #             in_building_roi = True
        #             building_roi_enter_time = current_timestamp
        #     else:
        #         if in_building_roi:
        #             # Exit building ROI
        #             in_building_roi = False
        #             total_gaze_duration += current_timestamp - building_roi_enter_time

        #     # Draw rectangular box around the building ROI
        #     cv2.rectangle(frame, (building_roi['x'], building_roi['y']),
        #                   (building_roi['x'] + building_roi['width'], building_roi['y'] + building_roi['height']),
        #                   (0, 255, 0), 2)


    # Draw circles at the gaze points if they are known
    # if x_left is not None and y_left is not None:
        cv2.circle(frame, (int(x_left_pixel), int(y_left_pixel)),circle_radius, (0, 0, 255), 13)
    # if x_right is not None and y_right is not None:
        cv2.circle(frame, (int(x_right_pixel), int(y_right_pixel)),circle_radius, (255, 0, 9), 13)
    # Increment frames_processed
        frames_processed += 1
    # Write the frame to the video writer
    out.write(frame)
    # Display the frame with gaze points
    # cv2.imshow('Gaze Overlay', frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    
# Print the total gaze duration on the building ROI
# print(f"Total Gaze Duration on Building ROI (2 minutes to 2.30 minutes): {total_gaze_duration} seconds")

# Release video writer and capture objects
out.release()
video_capture.release()
cv2.destroyAllWindows()

# Load the processed video with gaze points
processed_video_clip = VideoFileClip(temp_path)

# Load the original audio from the original video
original_audio_clip = VideoFileClip(video_file).audio

# Trim the original audio based on the provided start and end times
trimmed_audio_clip = original_audio_clip.subclip(minutes_to_skip * 60, end_time * 60)

# Set the original audio to the processed video
processed_video_clip = processed_video_clip.set_audio(trimmed_audio_clip)

# Write the final video with synchronized audio
processed_video_clip.write_videofile(destination_path, codec='libx264', audio_codec='aac', temp_audiofile='temp.m4a', remove_temp=True, fps=fps)

# Close the processed video clip
processed_video_clip.reader.close()
