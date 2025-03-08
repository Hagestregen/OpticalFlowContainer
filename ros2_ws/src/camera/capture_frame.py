#!/usr/bin/env python3

import pyrealsense2 as rs
import cv2
import os
import datetime

def capture_and_save_image(save_folder="realsense_images"):
    """
    Captures a single RGB image from a RealSense camera, saves it to a folder, and exits.
    
    Args:
        save_folder (str): Directory to save the image. Creates it if it doesn't exist.
    """
    # Create the save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Initialize the RealSense pipeline
    pipeline = rs.pipeline()
    
    # Configure the pipeline to stream color frames
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # RGB at 1280x720, 30 FPS
    
    try:
        # Start the pipeline
        pipeline.start(config)
        
        # Wait for the first frame
        frames = pipeline.wait_for_frames(timeout_ms=5000)  # Wait up to 5 seconds for a frame
        
        if frames:
            # Get the color frame
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("Error: No color frame received.")
                return
            
            # Convert the frame to a numpy array (OpenCV format)
            color_image = np.asanyarray(color_frame.get_data())
            
            # Generate a unique filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_folder, f"realsense_image_{timestamp}.jpg")
            
            # Save the image
            cv2.imwrite(filename, color_image)
            print(f"Image saved as: {filename}")
        else:
            print("Error: No frames received within timeout.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        # Stop the pipeline and clean up
        pipeline.stop()
        print("RealSense pipeline stopped. Exiting...")

if __name__ == "__main__":
    import numpy as np  # Import numpy here to avoid issues if not used earlier
    
    # Set the folder where images will be saved
    save_folder = "realsense_images"
    
    # Capture and save the image
    capture_and_save_image(save_folder)