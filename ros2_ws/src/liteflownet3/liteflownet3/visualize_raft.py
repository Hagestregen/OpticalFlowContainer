import os
import cv2  # OpenCV library for video and image processing
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend
import matplotlib.pyplot as plt  # For plotting and visualizing frames
from torchvision.transforms import functional as F
from torchvision.models.optical_flow import raft_large
import torch
import numpy as np


def compute_optical_flow(frame1, frame2, model, device):
    frame1_tensor = F.to_tensor(frame1).unsqueeze(0).to(device)     # Shape: [1, 3, H, W] - 3 is the RGB Channels
    frame2_tensor = F.to_tensor(frame2).unsqueeze(0).to(device)
    
    #Perform inference
    with torch.no_grad():
        flow = model(frame1_tensor, frame2_tensor)[0]    #Shape [1,2,H,W] - 2 is now the u and v -> (x,y) direction
        
    #Process flow results
    flow = flow[0] # Remove the batch dimension. Shape: [2, H, W] - 2 is the (u, v) dimension
    flow = flow.permute(1 ,2, 0).cpu().numpy()   # Convert to [H, W, 2] and NumPy format
    return flow


def visualize_optical_flow_arrow(optical_flow, start_frame, scale):
    # Convert flow values back to NumPy array
    flow = np.array(optical_flow)  # Shape: [H, W, 2]

    # Create a copy of the start frame for visualization with flow arrows
    vis_frame_with_arrows = start_frame.copy()

    H, W, _ = flow.shape # Get flow dimensions

    # Draw optical flow arrows on the start frame
    for y in range(0, H, 100):  # Sample every 100th pixel
        for x in range(0, W, 100):
            dx, dy = flow[y, x]  # Flow vector at (x, y)
            start_point = (x, y)
            end_point = (int(x + scale * dx), int(y + scale * dy)) # may adjust scale for better visualization
            cv2.arrowedLine(vis_frame_with_arrows, start_point, end_point, (0, 255, 0), thickness=2, tipLength=0.3)

    # Convert BGR to RGB for visualization with Matplotlib
    vis_frame_with_arrows_rgb = cv2.cvtColor(vis_frame_with_arrows, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 5))
    plt.imshow(vis_frame_with_arrows_rgb)
    plt.title("Arrow Head Optical Flow Visualization (Sampled Pixels)")
    plt.axis("off")
    plt.savefig('flow_plot.png')



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

# Path to the video file
video_path = 'output.avi'

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
model = raft_large(pretrained=True, progress=True).eval().to(device)

# Extract frames from the video
frame_rate, frame_count, frames = extract_frames(video_path)


# Retrieve the first and last frames from the extracted frames
first_frame = frames[0]
last_frame = frames[1]



# Convert frames to RGB format (OpenCV loads images in BGR format by default)
first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)  
last_frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)


flow = compute_optical_flow(first_frame, last_frame, model, device)

visualize_optical_flow_arrow(flow, first_frame, scale=1)

# Create a side-by-side plot of the first and last frames
plt.figure(figsize=(6, 5))

# Plot the first frame
plt.subplot(1, 2, 1)
plt.imshow(first_frame_rgb)  # Displays the first frame in RGB format
plt.title("First Frame")
plt.axis("off")  # Turn off axis ticks and labels for better visualization

# Plot the last frame
plt.subplot(1, 2, 2)
plt.imshow(last_frame_rgb)  
plt.title("Last Frame")
plt.axis("off")  

# Display the plot
plt.tight_layout()  # Adjusts the layout to avoid overlapping of titles and labels
plt.savefig('frames_plot.png')
plt.close()