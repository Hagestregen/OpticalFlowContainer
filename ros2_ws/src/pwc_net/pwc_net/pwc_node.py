#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import math
import threading
from collections import deque
from ament_index_python.packages import get_package_share_directory
import csv
import os
import time

from .pwc_net import Network  # This module now contains both the network definition and a global estimate() method

class PWCNetNode(Node):
    def __init__(self):
        super().__init__('pwc_net_direct_node')
        # Declare and retrieve parameters
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('pixel_to_meter', 0.000857)
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        self.pixel_to_meter = self.get_parameter('pixel_to_meter').value
        

        
        self.writeCsv = False
        
        if self.writeCsv:
            # Prepare CSV file for inference times
            self.csv_filename = f"pwc_direct_inference_{self.width}x{self.height}.csv"
            # Write header if new file
            if not os.path.isfile(self.csv_filename):
                with open(self.csv_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['timestamp', 'inference_time_s'])

        # Create publishers for raw and smoothed optical flow velocity
        self.flow_pub = self.create_publisher(Vector3Stamped, '/optical_flow/PWC_velocity', 10)
        self.smooth_flow_pub = self.create_publisher(Vector3Stamped, '/optical_flow/PWC_smooth_velocity', 10)

        self.get_logger().info('Initializing PWC-Net model...')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Instantiate network – note that Network loads its weights and offers estimate(…) in its definition.
        self.net = Network().to(self.device).eval()
        self.get_logger().info('PWC-Net model loaded.')

        # Variables for previous frame and for smoothing velocity estimates
        self.prev_tensor = None
        self.prev_time = None
        self.velocity_buffer = deque(maxlen=3)

        # Subscription for depth measurements (to update pixel_to_meter dynamically)
        self.median_depth = None
        self.sub_depth = self.create_subscription(
            Float32,
            '/camera/depth/median_distance',
            self.depth_callback,
            10
        )

        # Initialize RealSense pipeline in a separate thread
        self.pipeline = rs.pipeline()
        self.pipeline_thread = threading.Thread(target=self.run_pipeline_loop)
        self.pipeline_thread.daemon = True
        self.pipeline_thread.start()
        self.focal_length_x = None

    def depth_callback(self, msg):
        self.median_depth = msg.data
        if self.focal_length_x is not None:
            self.pixel_to_meter = self.median_depth / self.focal_length_x
            self.get_logger().info(f"Updated pixel_to_meter: {self.pixel_to_meter:.6f}")

    def run_pipeline_loop(self):
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.pipeline.start(config)
        self.get_logger().info(f"RealSense pipeline started: {self.width}x{self.height} at {self.fps} FPS.")
        
        # Retrieve intrinsics to update focal length and conversion factor later
        profile = self.pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.focal_length_x = intrinsics.fx
        self.get_logger().info(f"Focal length: {self.focal_length_x} pixels")
        
        frames0    = self.pipeline.wait_for_frames()
        color_frame0 = frames0.get_color_frame()
        rs_t0      = color_frame0.get_timestamp() / 1000.0        # device clock
        ros_t0_msg = self.get_clock().now().to_msg()              # ROS clock
        ros_t0     = ros_t0_msg.sec + ros_t0_msg.nanosec * 1e-9
        self.device_to_ros_offset = ros_t0 - rs_t0      
        
        try:
            while rclpy.ok():
                frames = self.pipeline.wait_for_frames()      # ROS time at roughly the same instant

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                rs_t = color_frame.get_timestamp() / 1000.0    # seconds on device clock
                ros_t = rs_t + self.device_to_ros_offset  
                ros_stamp = Time()
                sec  = int(ros_t)
                nsec = int((ros_t - sec) * 1e9)
                ros_stamp.sec = sec
                ros_stamp.nanosec = nsec

                
                # ros_stamp = self.get_clock().now().to_msg()
                color_image = np.asanyarray(color_frame.get_data())
                
                if self.writeCsv:
                    start_time = time.perf_counter()
                
                # Convert from BGR to RGB and normalize to [0,1]
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                current_tensor = torch.from_numpy(color_image).permute(2, 0, 1).float() / 255.0
                current_time = frames.get_timestamp() / 1000.0  # seconds

                # For the very first frame, just store and continue
                if self.prev_tensor is None:
                    self.prev_tensor = current_tensor
                    self.prev_time = current_time
                    continue

                dt = current_time - self.prev_time
                if dt <= 0:
                    dt = 1e-3
                self.prev_time = current_time

                try:
                    # Move the previous and current frames to device
                    tenOne = self.prev_tensor.to(self.device)
                    tenTwo = current_tensor.to(self.device)
                    # Use the network's estimate() method to compute optical flow.
                    # This method encapsulates resizing, forward pass, and scaling.
                    tenFlow = self.net.estimate(tenOne, tenTwo)
                    # tenFlow has shape (2, H, W) on CPU
                    flow_np = tenFlow.numpy()  # Convert to NumPy array
                    # Compute average horizontal flow (pixels/sec) and convert to m/s
                    # u_avg = np.mean(flow_np[0]) / dt
                    u_avg = np.median(flow_np[0]) / dt
                    vx_m_per_s = float(u_avg * self.pixel_to_meter)
                    self.velocity_buffer.append(vx_m_per_s)
                    smoothed_vx = float(np.mean(self.velocity_buffer))

                    # Publish raw optical flow velocity as a ROS message
                    vel_msg = Vector3Stamped()
                    # vel_msg.header.stamp = ros_stamp
                    vel_msg.header.stamp.sec       = int(ros_t)
                    vel_msg.header.stamp.nanosec   = int((ros_t - int(ros_t)) * 1e9)
                    vel_msg.header.frame_id = "camera_link"
                    vel_msg.vector.x = vx_m_per_s
                    vel_msg.vector.y = 0.0
                    vel_msg.vector.z = 0.0
                    self.flow_pub.publish(vel_msg)
                    self.get_logger().info(f"Flow velocity: {vel_msg.vector.x:.4f} m/s")

                    # Publish smoothed optical flow velocity
                    vel_smooth_msg = Vector3Stamped()
                    # vel_smooth_msg.header.stamp = ros_stamp
                    vel_msg.header.stamp.sec       = int(ros_t)
                    vel_msg.header.stamp.nanosec   = int((ros_t - int(ros_t)) * 1e9)
                    vel_smooth_msg.header.frame_id = "camera_link"
                    vel_smooth_msg.vector.x = smoothed_vx
                    vel_smooth_msg.vector.y = 0.0
                    vel_smooth_msg.vector.z = 0.0
                    self.smooth_flow_pub.publish(vel_smooth_msg)

                    self.prev_tensor = current_tensor  # Update for next iteration
                    
                    if self.writeCsv:
                        end_time = time.perf_counter()
                        inference_time = end_time - start_time    
                    
                        # Append to CSV   
                        timestamp = time.time()   
                        with open(self.csv_filename, 'a', newline='') as csvfile: 
                            csv.writer(csvfile).writerow([timestamp, inference_time]) 
                        
                except Exception as e:
                    self.get_logger().error(f"Error computing optical flow: {str(e)}")
        except Exception as e:
            self.get_logger().error(f"Error in pipeline loop: {str(e)}")
        finally:
            self.pipeline.stop()

def main(args=None):
    rclpy.init(args=args)
    node = PWCNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down pwc_net_node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()



