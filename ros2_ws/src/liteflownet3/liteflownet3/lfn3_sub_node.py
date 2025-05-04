#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32
from sensor_msgs.msg import Image, Range
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import math
import time
from collections import deque
import os
from ament_index_python.packages import get_package_share_directory
import pyrealsense2 as rs
import time
import csv


from .liteflownet3 import Network

class OpticalFlowDirectNode(Node):
    def __init__(self):
        super().__init__('optical_flow_direct_node')
        
        # Declare parameters
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('width_depth', 640)
        self.declare_parameter('height_depth', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('pixel_to_meter', 0.000857)  # default for 640x480
        
        # Retrieve parameter values
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.width_depth = self.get_parameter('width_depth').value
        self.height_depth = self.get_parameter('height_depth').value
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
        
        self.get_logger().info("Starting RealSense pipeline to read intrinsics…")
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        # match your ROS params here:
        cfg.enable_stream(rs.stream.color,
                          self.width, self.height,
                          rs.format.rgb8, self.fps)
        cfg.enable_stream(rs.stream.depth,
                          self.width_depth, self.height_depth,
                          rs.format.z16, self.fps)
        
        profile = self.pipeline.start(cfg)
        color_profile = profile.get_stream(rs.stream.color) \
                               .as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        # focal length in pixels:
        self.focal_length_x = intr.fx
        self.get_logger().info(f"RealSense fx = {self.focal_length_x:.2f} px")
        
        # Publisher for velocity
        self.flow_pub = self.create_publisher(Vector3Stamped, '/optical_flow/LFN3_velocity', 10)
        self.smooth_flow_pub = self.create_publisher(Vector3Stamped, '/optical_flow/LFN3_smooth_velocity', 10)
        
        # Initialize LiteFlowNet3 model
        self.get_logger().info('Initializing LiteFlowNet3 model...')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Network().to(self.device).eval()
        
        # Load weights
        package_share_dir = get_package_share_directory('liteflownet3')
        model_path = os.path.join(package_share_dir, 'network-sintel.pytorch')
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.get_logger().info('LiteFlowNet3 model loaded.')
        
        # Normalization tensors
        self.mean_one = torch.tensor([0.411618, 0.434631, 0.454253]).view(3,1,1).to(self.device)
        self.mean_two = torch.tensor([0.410782, 0.433645, 0.452793]).view(3,1,1).to(self.device)
        
        # State for optical flow
        self.prev_tensor = None
        self.prev_time = None
        self.velocity_buffer = deque(maxlen=5)
        
        # Depth subscription (for pixel_to_meter update)
        self.median_depth = None
        self.declare_parameter('camera_info_focal_x', self.focal_length_x)
        self.focal_length_x = self.get_parameter('camera_info_focal_x').value
        self.sub_depth = self.create_subscription(
            Range, '/camera/depth/median_distance',
            self.depth_callback, 10)
        
        # Image subscription instead of RealSense
        self.bridge = CvBridge()
        self.sub_image = self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self.image_callback, 10)
        self.get_logger().info(f"Subscribed to /camera/camera/color/image_raw at expected {self.width}×{self.height}")
        
    def depth_callback(self, msg: Range):
        depth = float(msg.range)
        self.median_depth = depth
        if self.focal_length_x:
            self.pixel_to_meter = self.median_depth / self.focal_length_x
            self.get_logger().info(f"Updated pixel_to_meter: {self.pixel_to_meter:.6f}")
    
    def image_callback(self, msg: Image):
        
        if self.writeCsv:
            start_time = time.perf_counter()
        
        # convert ROS Image to OpenCV BGR image
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return
        
        # resize if not matching expected resolution
        if (cv_bgr.shape[1] != self.width) or (cv_bgr.shape[0] != self.height):
            cv_bgr = cv2.resize(cv_bgr, (self.width, self.height))
        
        # BGR → RGB
        cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        current_tensor = torch.from_numpy(cv_rgb).permute(2,0,1).float() / 255.0
        
        # timestamp from header
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        if self.prev_tensor is None:
            self.prev_tensor = current_tensor
            self.prev_time = current_time
            return
        
        # time delta
        dt = current_time - self.prev_time
        if dt <= 0:
            dt = 1e-3
        self.prev_time = current_time
        
        try:
            # normalize
            tenOne = (self.prev_tensor.to(self.device) - self.mean_one)
            tenTwo = (current_tensor.to(self.device) - self.mean_two)
            
            # sizes for network
            intW, intH = self.width, self.height
            preW = int(math.ceil(math.ceil(intW/32.0)*32.0))
            preH = int(math.ceil(math.ceil(intH/32.0)*32.0))
            
            
            
            with torch.no_grad():
                t1 = torch.nn.functional.interpolate(
                    tenOne.unsqueeze(0), size=(preH,preW),
                    mode='bilinear', align_corners=False)
                t2 = torch.nn.functional.interpolate(
                    tenTwo.unsqueeze(0), size=(preH,preW),
                    mode='bilinear', align_corners=False)
                flow = self.net(t1,t2)
                flow = torch.nn.functional.interpolate(
                    flow, size=(intH,intW),
                    mode='bilinear', align_corners=False)
                flow[:,0,:,:] *= intW/preW
                flow[:,1,:,:] *= intH/preH
            
            
            # self.get_logger().info(f"Inference time: {inference_time:.6f} seconds")
            
            # average horizontal flow
            flow_np = flow[0].cpu().numpy()
            u_avg = np.mean(flow_np[0]) / dt
            vx = float(u_avg * self.pixel_to_meter)
            
            # smoothing
            self.velocity_buffer.append(vx)
            vx_smooth = float(np.mean(self.velocity_buffer))
            
            # publish raw and smooth
            for pub, vel in ((self.flow_pub, vx),(self.smooth_flow_pub, vx_smooth)):
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
    node = OpticalFlowDirectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down optical_flow_direct_node')
    finally:
        node.pipeline.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
