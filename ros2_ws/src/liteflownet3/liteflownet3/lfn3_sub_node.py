#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32
from sensor_msgs.msg import Image, Range, CameraInfo
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
        super().__init__('lfn3_sub_node')
        
        # Declare parameters
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('width_depth', 640)
        self.declare_parameter('height_depth', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('pixel_to_meter', 0.001100)  # placeholder gets updated in callback
        
        # Retrieve parameter values
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.width_depth = self.get_parameter('width_depth').value
        self.height_depth = self.get_parameter('height_depth').value
        self.fps = self.get_parameter('fps').value
        self.pixel_to_meter = self.get_parameter('pixel_to_meter').value
        
        
        self.declare_parameter('visualize', False)
        self.visualize = self.get_parameter('visualize').value
        self.declare_parameter('dense_viz', False)
        self.dense_viz = self.get_parameter('dense_viz').value
        self.declare_parameter('max_speed', 0.6)      # meters per second
        self.max_speed = self.get_parameter('max_speed').value



        
        self.writeCsv = False
        # self.focal_length_x = 417.19 
        self.focal_length_x = None
        
        if self.writeCsv:
            # Prepare CSV file for inference times
            self.csv_filename = f"lfn3_sub_inference_{self.width}x{self.height}.csv"
            # Write header if new file
            if not os.path.isfile(self.csv_filename):
                with open(self.csv_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['timestamp', 'inference_time_s'])
        
        # self.get_logger().info("Starting RealSense pipeline to read intrinsics…")
        # self.pipeline = rs.pipeline()
        # cfg = rs.config()
        # # match your ROS params here:
        # cfg.enable_stream(rs.stream.color,
        #                   self.width, self.height,
        #                   rs.format.rgb8, self.fps)
        # cfg.enable_stream(rs.stream.depth,
        #                   self.width_depth, self.height_depth,
        #                   rs.format.z16, self.fps)
        
        # profile = self.pipeline.start(cfg)
        # color_profile = profile.get_stream(rs.stream.color) \
        #                        .as_video_stream_profile()
        # intr = color_profile.get_intrinsics()
        # # focal length in pixels:
        # self.focal_length_x = intr.fx
        # self.get_logger().info(f"RealSense fx = {self.focal_length_x:.2f} px")
        
        # Publisher for velocity
        self.flow_pub = self.create_publisher(Vector3Stamped, '/optical_flow/LFN3_velocity', 10)
        self.smooth_flow_pub = self.create_publisher(Vector3Stamped, '/optical_flow/LFN3_smooth_velocity', 10)
        
        # Initialize LiteFlowNet3 model
        self.get_logger().info('Initializing LiteFlowNet3 model...')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")
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
        # self.declare_parameter('camera_info_focal_x', self.focal_length_x)
        # self.focal_length_x = self.get_parameter('camera_info_focal_x').value
        self.sub_depth = self.create_subscription(
            Range, '/camera/depth/median_distance',
            self.depth_callback, 10)
        
        self.sub_camera_info = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        
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
            # self.get_logger().info(f"Updated pixel_to_meter: {self.pixel_to_meter:.6f}")
            
    def camera_info_callback(self, msg: CameraInfo):
        if self.focal_length_x is None:
            self.focal_length_x = msg.k[0]  # fx is the first element in the K matrix
            self.get_logger().info(f"Received focal length fx = {self.focal_length_x:.2f} px")
    
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
            # u_avg = np.mean(flow_np[0]) / dt
            u_avg = np.median(flow_np[0]) / dt
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
                
            
            if self.visualize:
                # Visualize flow vectors on top of the original RGB image
                flow_vis = cv_rgb.copy()
                step = 20  # spacing between arrows
                flow_u = flow_np[0]  # horizontal
                flow_v = flow_np[1]  # vertical
                
                for y in range(0, self.height, step):
                    for x in range(0, self.width, step):
                        pt1 = (x, y)
                        dx = int(flow_u[y, x])
                        dy = int(flow_v[y, x])
                        pt2 = (x + dx, y + dy)
                        cv2.arrowedLine(flow_vis, pt1, pt2, (0, 255, 0), 1, tipLength=0.4)

                # Display with OpenCV
                cv2.imshow("LiteFlowNet3 Optical Flow", cv2.cvtColor(flow_vis, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                
            if self.dense_viz:
                # compute magnitude (pixels/frame) → magnitude_mps
                mag, ang = cv2.cartToPolar(flow_u, flow_v, angleInDegrees=False)
                mag_mps = (mag / dt) * self.pixel_to_meter

                # clamp into [0 … max_speed]
                mag_norm = np.clip(mag_mps / self.max_speed, 0.0, 1.0)

                # HSV: hue from angle, saturation full, value = mag_norm*255
                hsv = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                hsv[..., 0] = np.uint8((ang * 90.0 / np.pi))    # Hue
                hsv[..., 1] = 255                               # Saturation
                hsv[..., 2] = np.uint8(mag_norm * 255)          # Value scaled

                bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imshow("Dense Flow", bgr_flow)
                cv2.waitKey(1)

            
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
        # node.pipeline.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
