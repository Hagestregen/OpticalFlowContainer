#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Image, Range, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import math
from collections import deque
import pyrealsense2 as rs
import os
import csv
from ament_index_python.packages import get_package_share_directory
import time

from .pwc_net import Network  # network with estimate() method

class PWCNetNode(Node):
    def __init__(self):
        super().__init__('pwc_net_sub_node')
        # Declare parameters for image resolution, frame rate, and scaling
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('width_depth', 640)
        self.declare_parameter('height_depth', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('pixel_to_meter', 0.001100)
        
                # Declare display parameters
        self.declare_parameter('show_live_feed', True)
        self.declare_parameter('show_flow', True)
        self.declare_parameter('show_mask', True)
        
        
                # Publishers for visualization images
        self.live_feed_pub = self.create_publisher(Image, '/optical_flow/image_live_feed', 10)
        self.image_flow_pub = self.create_publisher(Image, '/optical_flow/image_flow', 10)
        self.image_mask_pub = self.create_publisher(Image, '/optical_flow/image_mask', 10)

        # Retrieve parameter values
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.width_depth = self.get_parameter('width_depth').value
        self.height_depth = self.get_parameter('height_depth').value
        self.fps = self.get_parameter('fps').value
        self.pixel_to_meter = self.get_parameter('pixel_to_meter').value
        
        self.writeCsv = False
        self.focal_length_x = None
        
        if self.writeCsv:
            # Prepare CSV file for inference times
            self.csv_filename = f"pwc_inference_{self.width}x{self.height}.csv"
            # Write header if new file
            if not os.path.isfile(self.csv_filename):
                with open(self.csv_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['timestamp', 'inference_time_s'])

        # # --- RealSense intrinsics only (no frame loop) ---
        # self.get_logger().info('Starting RealSense pipeline to read intrinsics…')
        # pipeline = rs.pipeline()
        # cfg = rs.config()
        # cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        # cfg.enable_stream(rs.stream.depth, self.width_depth, self.height_depth, rs.format.z16, self.fps)
        # profile = pipeline.start(cfg)
        # color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        # intr = color_profile.get_intrinsics()
        # self.focal_length_x = intr.fx
        # self.get_logger().info(f'RealSense fx = {self.focal_length_x:.2f} px')
        # pipeline.stop()
        # # ----------------------------------------------

        # Publishers for raw and smoothed optical flow
        self.flow_pub = self.create_publisher(Vector3Stamped,
                                              '/optical_flow/PWC_velocity', 10)
        self.smooth_flow_pub = self.create_publisher(Vector3Stamped,
                                                     '/optical_flow/PWC_smooth_velocity', 10)

        # Initialize PWC-Net
        self.get_logger().info('Initializing PWC-Net model...')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Network().to(self.device).eval()
        state_dict = torch.hub.load_state_dict_from_url(
            'http://content.sniklaus.com/github/pytorch-pwc/network-default.pytorch',
            file_name='pwc-default',
            map_location=self.device
        )
        # rename the keys so they match your module names
        state_dict = {
            k.replace('module', 'net'): v
            for k, v in state_dict.items()
        }
        self.net.load_state_dict(state_dict)
        self.get_logger().info("✅ Loaded PWC‑Net pretrained weights")
        self.get_logger().info('PWC-Net model loaded.')

        # State for optical flow and smoothing
        self.prev_tensor = None
        self.prev_time = None
        self.velocity_buffer = deque(maxlen=5)

        # Depth subscription (Range message)
        self.median_depth = None
        self.sub_depth = self.create_subscription(
            Range,
            '/camera/depth/median_distance',
            self.depth_callback,
            10
        )
        
        self.sub_camera_info = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)

        # Image subscription (use width/height parameters)
        self.bridge = CvBridge()
        self.sub_image = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        self.get_logger().info(
            f"Subscribed to /camera/color/image_raw at {self.width}×{self.height}@{self.fps}FPS"
        )

    def depth_callback(self, msg: Range):
        # Treat Range.range as depth in meters
        self.median_depth = float(msg.range)
        if self.focal_length_x:
            self.pixel_to_meter = self.median_depth / self.focal_length_x
            self.get_logger().info(
                f"Updated pixel_to_meter: {self.pixel_to_meter:.6f} m/px"
            )
            
    def camera_info_callback(self, msg: CameraInfo):
        if self.focal_length_x is None:
            self.focal_length_x = msg.k[0]  # fx is the first element in the K matrix
            self.get_logger().info(f"Received focal length fx = {self.focal_length_x:.2f} px")

    def image_callback(self, msg: Image):
        # Compute flow via PWC-Net's estimate()
        if self.writeCsv:
            start_time = time.perf_counter()
        # Convert ROS Image to OpenCV BGR image
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return

        # Resize if incoming image != expected resolution
        if (cv_bgr.shape[1] != self.width) or (cv_bgr.shape[0] != self.height):
            cv_bgr = cv2.resize(cv_bgr, (self.width, self.height))

        # BGR → RGB and to torch tensor
        # cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        current_tensor = torch.from_numpy(cv_bgr).permute(2,0,1).float() / 255.0

        # Timestamp from header
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # First frame: just store
        if self.prev_tensor is None:
            self.prev_tensor = current_tensor
            self.prev_time = current_time
            return

        # Compute dt
        dt = current_time - self.prev_time
        if dt <= 0:
            dt = 1e-3
        self.prev_time = current_time

        try:
            # Move to device
            tenOne = self.prev_tensor.to(self.device)
            tenTwo = current_tensor.to(self.device)

            with torch.no_grad():
                tenFlow = self.net.estimate(tenOne, tenTwo)  # shape (2,H,W)
            
            
            
            flow_np = tenFlow.cpu().numpy()

            # Average horizontal flow (px/sec) → m/s
            u_avg = np.mean(flow_np[0]) / dt
            vx = float(u_avg * self.pixel_to_meter)

            # Smoothing
            self.velocity_buffer.append(vx)
            vx_smooth = float(np.mean(self.velocity_buffer))

            # Publish raw & smooth
            for pub, vel in ((self.flow_pub, vx), (self.smooth_flow_pub, vx_smooth)):
                msg_out = Vector3Stamped()
                msg_out.header.stamp = msg.header.stamp
                msg_out.header.frame_id = 'camera_link'
                msg_out.vector.x = vel
                msg_out.vector.y = 0.0
                msg_out.vector.z = 0.0
                pub.publish(msg_out)
                # self.get_logger().info(f"flow: {vx:.6f} m/s, smooth: {vx_smooth:.6f} m/s")

            self.prev_tensor = current_tensor
            
            if self.writeCsv:
                end_time = time.perf_counter()
                inference_time = end_time - start_time
                
                # Append to CSV
                timestamp = time.time()
                with open(self.csv_filename, 'a', newline='') as csvfile:
                    csv.writer(csvfile).writerow([timestamp, inference_time])

        except Exception as e:
            self.get_logger().error(f"Error computing optical flow: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PWCNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down pwc_net_node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
