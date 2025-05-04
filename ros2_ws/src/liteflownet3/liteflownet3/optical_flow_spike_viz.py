#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import math
import time
import threading
from collections import deque
import os
from ament_index_python.packages import get_package_share_directory

from .liteflownet3 import Network

class OpticalFlowDirectNode(Node):
    def __init__(self):
        super().__init__('optical_flow_direct_node')
        
        # Declare parameters
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('pixel_to_meter', 0.000857)  # For 640x480
        self.declare_parameter('spike_threshold', 0.00075)  # Threshold for spike detection in m/s
        
        # Retrieve parameter values
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        self.pixel_to_meter = self.get_parameter('pixel_to_meter').value
        self.spike_threshold = self.get_parameter('spike_threshold').value
        
        # Publisher for velocity
        self.flow_pub = self.create_publisher(Vector3Stamped, '/optical_flow/LFN3_velocity', 10)
        self.smooth_flow_pub = self.create_publisher(Vector3Stamped, '/optical_flow/LFN3_smooth_velocity', 10)
        
        # Initialize LiteFlowNet3 model
        self.get_logger().info('Initializing LiteFlowNet3 model...')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Network().to(self.device).eval()
        
        # Use get_package_share_directory to locate the model file
        package_share_dir = get_package_share_directory('liteflownet3')
        model_path = os.path.join(package_share_dir, 'network-sintel.pytorch')
        
        # Load the model weights
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.get_logger().info('LiteFlowNet3 model loaded.')
        
        # Normalization tensors (verify these match LiteFlowNet3 training)
        self.mean_one = torch.tensor([0.411618, 0.434631, 0.454253]).view(3, 1, 1).to(self.device)
        self.mean_two = torch.tensor([0.410782, 0.433645, 0.452793]).view(3, 1, 1).to(self.device)
        
        # Variables for previous frame and smoothing
        self.prev_tensor = None
        self.prev_time = None
        self.velocity_buffer = deque(maxlen=5)
        
        # Subscription to median depth
        self.median_depth = None
        self.sub_depth = self.create_subscription(
            Float32,
            '/camera/depth/median_distance',
            self.depth_callback,
            10
        )
        
        # Initialize the RealSense pipeline
        self.pipeline = rs.pipeline()
        
        # Start RealSense pipeline in a thread
        self.pipeline_thread = threading.Thread(target=self.run_pipeline_loop)
        self.pipeline_thread.daemon = True
        self.pipeline_thread.start()
        
        self.focal_length_x = None
        
    def depth_callback(self, msg):
        self.median_depth = msg.data
        if self.focal_length_x is not None:
            self.pixel_to_meter = self.median_depth / self.focal_length_x
            self.get_logger().info(f"Updated pixel_to_meter: {self.pixel_to_meter:.6f}")
    
    def visualize_flow(self, flow, image, threshold_factor=2):
        """Visualize optical flow by drawing arrows where flow magnitude exceeds a statistical threshold."""
        flow_np = flow.cpu().numpy()  # [2, H, W]
        flow_np = np.transpose(flow_np, (1, 2, 0))  # [H, W, 2]
        magnitude = np.sqrt(flow_np[:, :, 0]**2 + flow_np[:, :, 1]**2)
        mean_mag = np.mean(magnitude)
        std_mag = np.std(magnitude)
        threshold = mean_mag + threshold_factor * std_mag
        mask = magnitude > threshold
        
        vis_image = image.copy()
        h, w = flow_np.shape[:2]
        step = 10  # Draw arrows every 10 pixels to avoid clutter
        for y in range(0, h, step):
            for x in range(0, w, step):
                if mask[y, x]:
                    start_point = (x, y)
                    dx = flow_np[y, x, 0]
                    dy = flow_np[y, x, 1]
                    end_point = (int(x + dx), int(y + dy))
                    cv2.arrowedLine(vis_image, start_point, end_point, (0, 255, 0), 1, tipLength=0.3)
        return vis_image
    
    def run_pipeline_loop(self):
        # Configure RealSense pipeline
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        self.pipeline.start(config)
        self.get_logger().info(f'RealSense pipeline started with {self.width}x{self.height} at {self.fps} FPS.')
        
        # Get focal length from color stream
        profile = self.pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.focal_length_x = intrinsics.fx
        self.get_logger().info(f"Focal length: {self.focal_length_x} pixels")
        
        # Create directory for saving spike images
        self.spike_image_dir = 'spike_images'
        os.makedirs(self.spike_image_dir, exist_ok=True)
        
        try:
            while rclpy.ok():
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                # Keep BGR image for visualization and convert to RGB for tensor
                color_image_bgr = np.asanyarray(color_frame.get_data())  # BGR
                color_image = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)  # RGB
                
                # Convert to tensor and normalize
                current_tensor = torch.from_numpy(color_image).permute(2, 0, 1).float() / 255.0
                current_time = frames.get_timestamp() / 1000.0  # ms to seconds
                
                # Skip first frame
                if self.prev_tensor is None:
                    self.prev_tensor = current_tensor
                    self.prev_time = current_time
                    continue
                
                # Time difference
                dt = current_time - self.prev_time
                if dt <= 0:
                    dt = 1e-3  # Prevent division by zero
                self.prev_time = current_time
                
                try:
                    # Preprocess tensors
                    tenOne = (self.prev_tensor.to(self.device) - self.mean_one)
                    tenTwo = (current_tensor.to(self.device) - self.mean_two)
                    
                    # Compute dimensions for resizing
                    intWidth = self.width
                    intHeight = self.height
                    intPreprocessedWidth = int(math.ceil(math.ceil(intWidth / 32.0) * 32.0))
                    intPreprocessedHeight = int(math.ceil(math.ceil(intHeight / 32.0) * 32.0))
                    
                    with torch.no_grad():
                        # Resize inputs
                        tenPreprocessedOne = torch.nn.functional.interpolate(
                            input=tenOne.unsqueeze(0),
                            size=(intPreprocessedHeight, intPreprocessedWidth),
                            mode='bilinear',
                            align_corners=False
                        )
                        tenPreprocessedTwo = torch.nn.functional.interpolate(
                            input=tenTwo.unsqueeze(0),
                            size=(intPreprocessedHeight, intPreprocessedWidth),
                            mode='bilinear',
                            align_corners=False
                        )
                        # Compute optical flow
                        tenFlow = self.net(tenPreprocessedOne, tenPreprocessedTwo)
                        # Resize flow back
                        tenFlow = torch.nn.functional.interpolate(
                            input=tenFlow,
                            size=(intHeight, intWidth),
                            mode='bilinear',
                            align_corners=False
                        )
                        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
                        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
                    
                    # Crop to upper two-thirds (optional, currently using full flow)
                    cropped_height = int(intHeight * 2 / 3)
                    tenFlow_cropped = tenFlow[:, :, :cropped_height, :]
                                        
                    # Compute velocity
                    flow_np = tenFlow[0].cpu().numpy()  # [2, H, W]
                    # flow_np = tenFlow_cropped[0].cpu().numpy()  # Uncomment to use cropped flow
                    u_avg = np.mean(flow_np[0]) / dt  # Horizontal flow (pixels/s)
                    vx_m_per_s = float(u_avg * self.pixel_to_meter)  # Meters/s
                    
                    # Smooth velocity
                    self.velocity_buffer.append(vx_m_per_s)
                    smoothed_vx = float(np.mean(self.velocity_buffer))
                    
                    # Publish velocity
                    vel_msg = Vector3Stamped()
                    vel_msg.header.stamp = self.get_clock().now().to_msg()
                    vel_msg.header.frame_id = "camera_link"
                    vel_msg.vector.x = vx_m_per_s
                    vel_msg.vector.y = 0.0  # Add vertical flow if needed
                    vel_msg.vector.z = 0.0
                    
                    vel_smooth_msg = Vector3Stamped()
                    vel_smooth_msg.header.stamp = self.get_clock().now().to_msg()
                    vel_smooth_msg.header.frame_id = "camera_link"
                    vel_smooth_msg.vector.x = smoothed_vx
                    vel_smooth_msg.vector.y = 0.0  # Add vertical flow if needed
                    vel_smooth_msg.vector.z = 0.0
                    
                    self.flow_pub.publish(vel_msg)
                    self.get_logger().info(f"flow is: {vel_msg.vector.x}")
                    self.smooth_flow_pub.publish(vel_smooth_msg)
                    
                    # Check for spikes and visualize
                    if abs(vx_m_per_s) > self.spike_threshold:
                        self.get_logger().warn(f"Spike detected: {vx_m_per_s} m/s")
                        flow_vis = self.visualize_flow(tenFlow[0], color_image_bgr)
                        timestamp = time.time()
                        filename = os.path.join(self.spike_image_dir, f'spike_{timestamp}.png')
                        cv2.imwrite(filename, flow_vis)
                        self.get_logger().info(f"Saved spike image: {filename}")
                    
                    # Update previous tensor
                    self.prev_tensor = current_tensor
                
                except Exception as e:
                    self.get_logger().error(f'Error computing optical flow: {str(e)}')
        
        except Exception as e:
            self.get_logger().error(f'Error in pipeline loop: {str(e)}')
        finally:
            self.pipeline.stop()

def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowDirectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down optical_flow_direct_node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()