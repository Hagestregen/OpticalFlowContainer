#!/usr/bin/env python3

import torch
import cv2
import numpy as np
import os
import math
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'correlation_package'))

# Set device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the LiteFlowNet3 model
net = Network().to(device).eval()
model_path = 'network-sintel.pytorch'  # Pre-trained model file in current directory
net.load_state_dict(torch.load(model_path, map_location=device))
print("LiteFlowNet3 model loaded successfully.")

# Define normalization means (as per LiteFlowNet3 requirements)
mean_one = torch.tensor([0.411618, 0.434631, 0.454253]).view(3, 1, 1).to(device)
mean_two = torch.tensor([0.410782, 0.433645, 0.452793]).view(3, 1, 1).to(device)

# Specify image paths
image1_path = 'images/image1.png'
image2_path = 'images/image2.png'

# Read the images
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Check if images were loaded successfully
if image1 is None or image2 is None:
    print("Error: Could not read one or both images. Ensure 'images/image1.png' and 'images/image2.png' exist.")
    exit()

# Verify that images have the same dimensions
if image1.shape != image2.shape:
    print("Error: Images must have the same dimensions.")
    exit()

# Get image dimensions
height, width = image1.shape[:2]

# Compute preprocessed dimensions (multiples of 32 for LiteFlowNet3)
intPreprocessedWidth = ((width + 31) // 32) * 32
intPreprocessedHeight = ((height + 31) // 32) * 32

# Preprocess images: Convert BGR to RGB, normalize, and convert to tensors
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image1_float = image1_rgb.astype(np.float32) / 255.0
image2_float = image2_rgb.astype(np.float32) / 255.0
tensor1 = torch.from_numpy(image1_float).permute(2, 0, 1).to(device)
tensor2 = torch.from_numpy(image2_float).permute(2, 0, 1).to(device)

# Apply normalization by subtracting mean tensors
tenOne = tensor1 - mean_one
tenTwo = tensor2 - mean_two

# Resize tensors to preprocessed dimensions
tenPreprocessedOne = torch.nn.functional.interpolate(
    tenOne.unsqueeze(0),
    size=(intPreprocessedHeight, intPreprocessedWidth),
    mode='bilinear',
    align_corners=False
)
tenPreprocessedTwo = torch.nn.functional.interpolate(
    tenTwo.unsqueeze(0),
    size=(intPreprocessedHeight, intPreprocessedWidth),
    mode='bilinear',
    align_corners=False
)

# Compute optical flow with LiteFlowNet3
with torch.no_grad():
    tenFlow = net(tenPreprocessedOne, tenPreprocessedTwo)

# Resize flow back to original dimensions
tenFlow = torch.nn.functional.interpolate(
    tenFlow,
    size=(height, width),
    mode='bilinear',
    align_corners=False
)

# Scale flow values based on resizing factors
tenFlow[:, 0, :, :] *= float(width) / float(intPreprocessedWidth)  # Horizontal scaling
tenFlow[:, 1, :, :] *= float(height) / float(intPreprocessedHeight)  # Vertical scaling

# Convert flow tensor to numpy array
flow_np = tenFlow[0].cpu().numpy()  # Shape: [2, H, W], where 0 is u (horizontal), 1 is v (vertical)

# Calculate average pixel displacements
u_avg = np.mean(flow_np[0])  # Average horizontal displacement
v_avg = np.mean(flow_np[1])  # Average vertical displacement
mag = np.sqrt(flow_np[0]**2 + flow_np[1]**2)  # Displacement magnitude per pixel
mag_avg = np.mean(mag)  # Average displacement magnitude

# Output results
print(f"Average horizontal displacement: {u_avg:.2f} pixels")
print(f"Average vertical displacement: {v_avg:.2f} pixels")
print(f"Average displacement magnitude: {mag_avg:.2f} pixels")