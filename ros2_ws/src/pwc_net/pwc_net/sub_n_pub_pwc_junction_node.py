#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Image, Range, PointCloud, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import math
from collections import deque
import os
import csv
import time
import message_filters
from ament_index_python.packages import get_package_share_directory

from .pwc_net import Network

class PWCNetNode(Node):
    def __init__(self):
        super().__init__('pwc_net_node')
        # Declare parameters
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('width_depth', 640)
        self.declare_parameter('height_depth', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('pixel_to_meter', 0.001100)
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
        self.focal_length_x = None

        if self.writeCsv:
            self.csv_filename = f"pwc_junction_{self.width}x{self.height}.csv"
            if not os.path.isfile(self.csv_filename):
                with open(self.csv_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['timestamp', 'inference_time_s'])

        # Publishers
        self.flow_pub = self.create_publisher(Vector3Stamped, '/optical_flow/PWC_velocity', 10)
        self.smooth_flow_pub = self.create_publisher(Vector3Stamped, '/optical_flow/PWC_smooth_velocity', 10)
        self.live_feed_pub = self.create_publisher(Image, '/optical_flow/image_live_feed', 10)
        self.image_flow_pub = self.create_publisher(Image, '/optical_flow/image_flow', 10)
        self.image_mask_pub = self.create_publisher(Image, '/optical_flow/image_mask', 10)

        # Initialize PWC-Net
        self.get_logger().info('Initializing PWC-Net model...')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Network().to(self.device).eval()
        state_dict = torch.hub.load_state_dict_from_url(
            'http://content.sniklaus.com/github/pytorch-pwc/network-default.pytorch',
            file_name='pwc-default',
            map_location=self.device
        )
        state_dict = {k.replace('module', 'net'): v for k, v in state_dict.items()}
        self.net.load_state_dict(state_dict)
        self.get_logger().info("✅ Loaded PWC‑Net pretrained weights")
        self.get_logger().info('PWC-Net model loaded.')

        # State variables
        self.prev_tensor = None
        self.prev_time = None
        self.velocity_buffer = deque(maxlen=5)

        # Subscriptions
        self.median_depth = None
        self.sub_depth = self.create_subscription(Range, '/camera/depth/median_distance', self.depth_callback, 10)
        self.sub_camera_info = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        self.bridge = CvBridge()
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.junction_sub = message_filters.Subscriber(self, PointCloud, '/junction_detector/junctions')
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.junction_sub], queue_size=10, slop=0.01)
        self.ts.registerCallback(self.synced_callback)
        self.get_logger().info("Subscribed to /camera/camera/color/image_raw and /junction_detector/junctions")

    def depth_callback(self, msg: Range):
        self.median_depth = float(msg.range)
        if self.focal_length_x:
            self.pixel_to_meter = self.median_depth / self.focal_length_x
            self.get_logger().info(f"Updated pixel_to_meter: {self.pixel_to_meter:.6f}")

    def camera_info_callback(self, msg: CameraInfo):
        if self.focal_length_x is None:
            self.focal_length_x = msg.k[0]
            self.get_logger().info(f"Received focal length fx = {self.focal_length_x:.2f} px")

    def flow_to_color(self, flow):
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

        current_tensor = torch.from_numpy(cv_bgr).permute(2, 0, 1).float() / 255.0
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
            tenOne = self.prev_tensor.to(self.device)
            tenTwo = current_tensor.to(self.device)
            with torch.no_grad():
                tenFlow = self.net.estimate(tenOne, tenTwo)
            flow_np = tenFlow.cpu().numpy()

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
                u_avg = np.mean(u_masked) / dt  # Switch to np.median() if desired
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

            if self.show_live_feed:
                image_with_junctions = cv_bgr.copy()
                for p in junctions:
                    cv2.circle(image_with_junctions, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
                ros_image = self.bridge.cv2_to_imgmsg(image_with_junctions, encoding="bgr8")
                ros_image.header = image_msg.header
                self.live_feed_pub.publish(ros_image)

            if self.show_flow:
                flow_vis = self.flow_to_color(flow_np)
                ros_flow = self.bridge.cv2_to_imgmsg(flow_vis, encoding="bgr8")
                ros_flow.header = image_msg.header
                self.image_flow_pub.publish(ros_flow)

            if self.show_mask:
                mask_vis = np.zeros_like(cv_bgr)
                mask_vis[mask] = [0, 255, 0]
                blended = cv2.addWeighted(cv_bgr, 0.7, mask_vis, 0.3, 0)
                ros_mask = self.bridge.cv2_to_imgmsg(blended, encoding="bgr8")
                ros_mask.header = image_msg.header
                self.image_mask_pub.publish(ros_mask)

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