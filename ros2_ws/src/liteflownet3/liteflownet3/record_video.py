## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)  # read video files
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # retrieves the video frame rate
    frames = []  # List to store extracted frames 
    success, frame_count = True, 0

    while success:
        success, frame = cap.read()  # Read the next frame; success indicates if reading was successful
        if not success:
            break
        frames.append(frame)  # Append the frame (NumPy array) to the list of frames
        frame_count += 1

    cap.release()  # Release the video capture object to free up resources
    return frame_rate, frame_count, frames

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))

frame_count = 0

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Write the frame to the video file
        out.write(color_image)

        # Increment frame counter and check if 5 seconds (150 frames) have been recorded
        frame_count += 1
        if frame_count >= 150:
            break

        # Display the stream (optional)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

    # Release the video writer after recording
    out.release()

finally:
    # Stop streaming
    pipeline.stop()