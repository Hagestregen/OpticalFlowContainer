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
        self.declare_parameter('pixel_to_meter', 0.001100)
        self.declare_parameter('visualize', True)
        self.declare_parameter('dense_viz', True)
        self.declare_parameter('max_speed', 0.6)
        
        # CLAHE parameters
        self.declare_parameter('apply_CLAHE', True)
        self.declare_parameter('clip_min', 1.0)
        self.declare_parameter('clip_max', 4.0)
        self.declare_parameter('C_min', 5.0)
        self.declare_parameter('C_max', 50.0)
        self.declare_parameter('tile_grid', [8, 8])

        # New parameters for additional preprocessing and post-processing
        self.declare_parameter('apply_bilateral_filter', False)
        self.declare_parameter('bilateral_d', 9)
        self.declare_parameter('bilateral_sigma_color', 75.0)
        self.declare_parameter('bilateral_sigma_space', 75.0)
        self.declare_parameter('apply_intensity_mask', False)
        self.declare_parameter('intensity_threshold', 200)
        self.declare_parameter('flow_magnitude_threshold', 0.1)
        self.declare_parameter('apply_median_filter', False)
        self.declare_parameter('median_kernel_size', 5)

        # Retrieve parameter values
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.width_depth = self.get_parameter('width_depth').value
        self.height_depth = self.get_parameter('height_depth').value
        self.fps = self.get_parameter('fps').value
        self.pixel_to_meter = self.get_parameter('pixel_to_meter').value
        self.visualize = self.get_parameter('visualize').value
        self.dense_viz = self.get_parameter('dense_viz').value
        self.max_speed = self.get_parameter('max_speed').value
        
        # CLAHE setup
        self.apply_CLAHE = self.get_parameter('apply_CLAHE').value
        self.clip_min = self.get_parameter('clip_min').value
        self.clip_max = self.get_parameter('clip_max').value
        self.C_min = self.get_parameter('C_min').value
        self.C_max = self.get_parameter('C_max').value
        grid = tuple(self.get_parameter('tile_grid').value)
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_min, tileGridSize=grid)

        # Retrieve new parameters
        self.apply_bilateral_filter = self.get_parameter('apply_bilateral_filter').value
        self.bilateral_d = self.get_parameter('bilateral_d').value
        self.bilateral_sigma_color = self.get_parameter('bilateral_sigma_color').value
        self.bilateral_sigma_space = self.get_parameter('bilateral_sigma_space').value
        self.apply_intensity_mask = self.get_parameter('apply_intensity_mask').value
        self.intensity_threshold = self.get_parameter('intensity_threshold').value
        self.flow_magnitude_threshold = self.get_parameter('flow_magnitude_threshold').value
        self.apply_median_filter = self.get_parameter('apply_median_filter').value
        self.median_kernel_size = self.get_parameter('median_kernel_size').value

        self.writeCsv = False
        self.focal_length_x = None
        
        if self.writeCsv:
            self.csv_filename = f"lfn3_sub_inference_{self.width}x{self.height}.csv"
            if not os.path.isfile(self.csv_filename):
                with open(self.csv_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['timestamp', 'inference_time_s'])
        
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
        
        # Depth subscription
        self.median_depth = None
        self.sub_depth = self.create_subscription(
            Range, '/camera/depth/median_distance',
            self.depth_callback, 10)
        
        self.sub_camera_info = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        
        # Image subscription
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
            
    def camera_info_callback(self, msg: CameraInfo):
        if self.focal_length_x is None:
            self.focal_length_x = msg.k[0]
            self.get_logger().info(f"Received focal length fx = {self.focal_length_x:.2f} px")
    
    def image_callback(self, msg: Image):
        if self.writeCsv:
            start_time = time.perf_counter()
        
        # Convert ROS Image to OpenCV BGR image
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return
        
        # Resize if not matching expected resolution
        if (cv_bgr.shape[1] != self.width) or (cv_bgr.shape[0] != self.height):
            cv_bgr = cv2.resize(cv_bgr, (self.width, self.height))
        
        # Apply adaptive CLAHE if enabled
        if self.apply_CLAHE:
            # Convert to HSV
            cv_hsv = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(cv_hsv)
            
            # Compute contrast from V channel
            contrast = np.std(v) / (np.mean(v) + 1e-3)
            clip = np.clip(
                self.clip_min + (contrast - self.C_min) /
                (self.C_max - self.C_min) * (self.clip_max - self.clip_min),
                self.clip_min, self.clip_max)
            self.clahe.setClipLimit(clip)
            
            # Apply CLAHE to V channel
            v_enhanced = self.clahe.apply(v)
            
            # Merge and convert back to RGB
            cv_hsv_enhanced = cv2.merge((h, s, v_enhanced))
            cv_rgb = cv2.cvtColor(cv_hsv_enhanced, cv2.COLOR_HSV2RGB)
        else:
            cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        
        # Apply bilateral filter if enabled
        if self.apply_bilateral_filter:
            cv_rgb = cv2.bilateralFilter(cv_rgb, d=self.bilateral_d,
                                         sigmaColor=self.bilateral_sigma_color,
                                         sigmaSpace=self.bilateral_sigma_space)
        
        current_tensor = torch.from_numpy(cv_rgb).permute(2,0,1).float() / 255.0
        
        # Timestamp from header
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        if self.prev_tensor is None:
            self.prev_tensor = current_tensor
            self.prev_time = current_time
            return
        
        # Time delta
        dt = current_time - self.prev_time
        if dt <= 0:
            dt = 1e-3
        self.prev_time = current_time
        
        try:
            # Normalize
            tenOne = (self.prev_tensor.to(self.device) - self.mean_one)
            tenTwo = (current_tensor.to(self.device) - self.mean_two)
            
            # Sizes for network
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
            
            # Convert flow to numpy
            flow_np = flow[0].cpu().numpy()
            
            # Apply median filter if enabled
            if self.apply_median_filter:
                flow_np[0] = cv2.medianBlur(flow_np[0], self.median_kernel_size)
                flow_np[1] = cv2.medianBlur(flow_np[1], self.median_kernel_size)
            
            # Apply magnitude threshold
            flow_magnitude = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
            mask_magnitude = flow_magnitude >= self.flow_magnitude_threshold
            flow_np[0] = flow_np[0] * mask_magnitude.astype(np.float32)
            flow_np[1] = flow_np[1] * mask_magnitude.astype(np.float32)
            
            # Apply intensity mask if enabled
            if self.apply_intensity_mask:
                gray = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2GRAY)
                mask_intensity = (gray < self.intensity_threshold).astype(np.float32)
                flow_np[0] = flow_np[0] * mask_intensity
                flow_np[1] = flow_np[1] * mask_intensity
            
            # Compute average horizontal flow
            u_avg = np.mean(flow_np[0]) / dt
            vx = float(u_avg * self.pixel_to_meter)
            
            # Smoothing
            self.velocity_buffer.append(vx)
            vx_smooth = float(np.mean(self.velocity_buffer))
            
            # Publish raw and smooth velocity
            for pub, vel in ((self.flow_pub, vx),(self.smooth_flow_pub, vx_smooth)):
                msg_out = Vector3Stamped()
                msg_out.header.stamp = msg.header.stamp
                msg_out.header.frame_id = 'camera_link'
                msg_out.vector.x = vel
                msg_out.vector.y = 0.0
                msg_out.vector.z = 0.0
                pub.publish(msg_out)
                
            if self.visualize:
                flow_vis = cv_rgb.copy()
                step = 20
                flow_u = flow_np[0]
                flow_v = flow_np[1]
                
                for y in range(0, self.height, step):
                    for x in range(0, self.width, step):
                        pt1 = (x, y)
                        dx = int(flow_u[y, x])
                        dy = int(flow_v[y, x])
                        pt2 = (x + dx, y + dy)
                        cv2.arrowedLine(flow_vis, pt1, pt2, (0, 255, 0), 1, tipLength=0.4)

                cv2.imshow("LiteFlowNet3 Optical Flow", cv2.cvtColor(flow_vis, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                
            if self.dense_viz:
                mag, ang = cv2.cartToPolar(flow_u, flow_v, angleínDegrees=False)
                mag_mps = (mag / dt) * self.pixel_to_meter
                mag_norm = np.clip(mag_mps / self.max_speed, 0.0, 1.0)

                hsv = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                hsv[..., 0] = np.uint8((ang * 90.0 / np.pi))
                hsv[..., 1] = 255
                hsv[..., 2] = np.uint8(mag_norm * 255)

                bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imshow("Dense Flow", bgr_flow)
                cv2.waitKey(1)
            
            self.prev_tensor = current_tensor
            
            if self.writeCsv:
                end_time = time.perf_counter()
                inference_time = end_time - start_time
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
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()