#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32
from sensor_msgs.msg import Image, Range, PointCloud, CameraInfo
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
import csv
import message_filters

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
        
        # Declare display parameters
        self.declare_parameter('show_live_feed', True)
        self.declare_parameter('show_flow', True)
        self.declare_parameter('show_mask', True)
        
        # Retrieve parameter values
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.width_depth = self.get_parameter('width_depth').value
        self.height_depth = self.get_parameter('height_depth').value
        self.fps = self.get_parameter('fps').value
        self.pixel_to_meter = self.get_parameter('pixel_to_meter').value
        self.show_live_feed = self.get_parameter('show_live_feed').value
        self.show_flow = self.get_parameter('show_flow').value
        self.show_mask = self.get_parameter('show_mask').value
        
        self.writeCsv = True
        
        if self.writeCsv:
            self.csv_filename = f"lfn3_inference_ junction_{self.width}x{self.height}.csv"
            if not os.path.isfile(self.csv_filename):
                with open(self.csv_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['timestamp', 'inference_time_s'])
        
        # self.get_logger().info("Starting RealSense pipeline to read intrinsics…")
        # self.pipeline = rs.pipeline()
        # cfg = rs.config()
        # cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        # cfg.enable_stream(rs.stream.depth, self.width_depth, self.height_depth, rs.format.z16, self.fps)
        
        # profile = self.pipeline.start(cfg)
        # color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        # intr = color_profile.get_intrinsics()
        # self.focal_length_x = intr.fx
        # self.get_logger().info(f"RealSense fx = {self.focal_length_x:.2f} px")
        
        # Publishers for velocity
        self.flow_pub = self.create_publisher(Vector3Stamped, '/optical_flow/LFN3_velocity', 10)
        self.smooth_flow_pub = self.create_publisher(Vector3Stamped, '/optical_flow/LFN3_smooth_velocity', 10)
        
        # Publishers for visualization images
        self.live_feed_pub = self.create_publisher(Image, '/optical_flow/image_live_feed', 10)
        self.image_flow_pub = self.create_publisher(Image, '/optical_flow/image_flow', 10)
        self.image_mask_pub = self.create_publisher(Image, '/optical_flow/image_mask', 10)
        
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
        
        # State variables
        self.prev_tensor = None
        self.prev_time = None
        self.velocity_buffer = deque(maxlen=5)
        
        # Depth subscription
        self.median_depth = None
        # self.declare_parameter('camera_info_focal_x', self.focal_length_x)
        # self.focal_length_x = self.get_parameter('camera_info_focal_x').value
        self.focal_length_x = None
        self.sub_depth = self.create_subscription(
            Range, '/camera/depth/median_distance',
            self.depth_callback, 10)
        
        # Synchronized image and junction subscriptions
        self.bridge = CvBridge()
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.junction_sub = message_filters.Subscriber(self, PointCloud, '/junction_detector/junctions')
        self.sub_camera_info = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.junction_sub], queue_size=10, slop=0.01)
        self.ts.registerCallback(self.synced_callback)
        self.get_logger().info(f"Subscribed to /camera/camera/color/image_raw and /junction_detector/junctions")
    
    def depth_callback(self, msg: Range):
        depth = float(msg.range)
        self.median_depth = depth
        if self.focal_length_x:
            self.pixel_to_meter = self.median_depth / self.focal_length_x
            self.get_logger().info(f"Updated pixel_to_meter: {self.pixel_to_meter:.6f}")
            
    def camera_info_callback(self, msg: CameraInfo):
        if self.focal_length_x is None:
            self.focal_length_x = msg.k[0]  # fx is the first element in the K matrix
            self.get_logger().info(f"Received focal length fx = {self.focal_length_x:.2f} px")
    
    def flow_to_color(self, flow):
        """Convert optical flow to a color visualization."""
        h, w = flow.shape[1:]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[0], flow[1])
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def synced_callback(self, image_msg, junction_msg):
        if self.writeCsv:
            start_time = time.perf_counter()
        
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return
        
        if (cv_bgr.shape[1] != self.width) or (cv_bgr.shape[0] != self.height):
            cv_bgr = cv2.resize(cv_bgr, (self.width, self.height))
        
        cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        current_tensor = torch.from_numpy(cv_rgb).permute(2,0,1).float() / 255.0
        
        junctions = [[p.x, p.y] for p in junction_msg.points]
        
        current_time = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9
        
        if self.prev_tensor is None:
            self.prev_tensor = current_tensor
            self.prev_time = current_time
            return
        
        dt = current_time - self.prev_time
        if dt <= 0:
            dt = 1e-3
        self.prev_time = current_time
        
        try:
            tenOne = (self.prev_tensor.to(self.device) - self.mean_one)
            tenTwo = (current_tensor.to(self.device) - self.mean_two)
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
            
            flow_np = flow[0].cpu().numpy()
            
            mask = np.zeros((self.height, self.width), dtype=bool)
            radius = 5
            for p in junctions:
                x, y = int(p[0]), int(p[1])
                if 0 <= x < self.width and 0 <= y < self.height:
                    y_min = max(0, y - radius)
                    y_max = min(self.height, y + radius + 1)
                    x_min = max(0, x - radius)
                    x_max = min(self.width, x + radius + 1)
                    mask[y_min:y_max, x_min:x_max] = True
            
            u_masked = flow_np[0][mask]
            if len(u_masked) > 0:
                # u_avg = np.mean(u_masked) / dt
                u_avg = np.median(u_masked) / dt
                vx = float(u_avg * self.pixel_to_meter)
                
                self.velocity_buffer.append(vx)
                vx_smooth = float(np.mean(self.velocity_buffer))
                
                for pub, vel in ((self.flow_pub, vx), (self.smooth_flow_pub, vx_smooth)):
                    msg_out = Vector3Stamped()
                    msg_out.header.stamp = image_msg.header.stamp
                    msg_out.header.frame_id = 'camera_link'
                    msg_out.vector.x = vel
                    msg_out.vector.y = 0.0
                    msg_out.vector.z = 0.0
                    pub.publish(msg_out)
            else:
                self.get_logger().warn("No valid regions to compute flow")
            
            # Visualization and publishing based on parameters
            if self.show_live_feed:
                image_with_junctions = cv_bgr.copy()
                for p in junctions:
                    cv2.circle(image_with_junctions, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
                # cv2.imshow('Live Feed with Junctions', image_with_junctions)
                # Publish the image
                ros_image = self.bridge.cv2_to_imgmsg(image_with_junctions, encoding="bgr8")
                ros_image.header = image_msg.header
                self.live_feed_pub.publish(ros_image)
            
            if self.show_flow:
                flow_vis = self.flow_to_color(flow_np)
                # cv2.imshow('Dense Optical Flow', flow_vis)
                # Publish the flow image
                ros_flow = self.bridge.cv2_to_imgmsg(flow_vis, encoding="bgr8")
                ros_flow.header = image_msg.header
                self.image_flow_pub.publish(ros_flow)
            
            if self.show_mask:
                mask_vis = np.zeros_like(cv_bgr)
                mask_vis[mask] = [0, 255, 0]
                blended = cv2.addWeighted(cv_bgr, 0.7, mask_vis, 0.3, 0)
                # cv2.imshow('Image with Mask', blended)
                # Publish the mask image
                ros_mask = self.bridge.cv2_to_imgmsg(blended, encoding="bgr8")
                ros_mask.header = image_msg.header
                self.image_mask_pub.publish(ros_mask)
            
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
        node.pipeline.stop()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
