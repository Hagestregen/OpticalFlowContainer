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
import time
import threading
from queue import Queue, Full
from collections import deque
import os
from ament_index_python.packages import get_package_share_directory
import csv
from rclpy.executors import MultiThreadedExecutor

from .liteflownet3 import Network

class OpticalFlowDirectNode(Node):
    def __init__(self):
        super().__init__('optical_flow_direct_node')
        
        # --- parameters ---
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('pixel_to_meter', 0.000857)
        self.width          = self.get_parameter('width').value
        self.height         = self.get_parameter('height').value
        self.fps            = self.get_parameter('fps').value
        self.pixel_to_meter = self.get_parameter('pixel_to_meter').value
        
        # optional CSV logging
        self.writeCsv = False
        if self.writeCsv:
            self.csv_filename = f"lfn3_direct_inference_{self.width}x{self.height}.csv"
            if not os.path.isfile(self.csv_filename):
                with open(self.csv_filename, 'w', newline='') as f:
                    csv.writer(f).writerow(['timestamp','inference_time_s'])
        
        # --- publishers ---
        self.flow_pub    = self.create_publisher(Vector3Stamped, '/optical_flow/LFN3_velocity',        10)
        self.smooth_pub  = self.create_publisher(Vector3Stamped, '/optical_flow/LFN3_smooth_velocity', 10)

        # --- LiteFlowNet3 init ---
        self.get_logger().info('Loading LiteFlowNet3…')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net    = Network().to(self.device).eval()
        pkg_dir     = get_package_share_directory('liteflownet3')
        path        = os.path.join(pkg_dir,'network-sintel.pytorch')
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.get_logger().info('Model ready.')
        
        # normalization constants
        self.mean_one = torch.tensor([0.411618,0.434631,0.454253]).view(3,1,1).to(self.device)
        self.mean_two = torch.tensor([0.410782,0.433645,0.452793]).view(3,1,1).to(self.device)
        
        # smoothing buffer
        self.prev_tensor     = None
        self.velocity_buffer = deque(maxlen=3)
        
        # depth → pixel_to_meter
        self.median_depth = None
        self.create_subscription(
            Float32,'/camera/depth/median_distance',
            self.depth_callback,10
        )
        
        # RealSense setup
        self.pipeline = rs.pipeline()
        
        # queue frames for inference
        self.frame_queue = Queue(maxsize=2)
        # threading.Thread(target=self._inference_worker,daemon=True).start()
        # threading.Thread(target=self.run_pipeline_loop, daemon=True).start()
        
        self.shutdown_event = threading.Event()
        
        self.inference_thread = threading.Thread(
        target=self._inference_worker, daemon=True)
        self.pipeline_thread  = threading.Thread(
        target=self.run_pipeline_loop, daemon=True)
        self.inference_thread.start()
        self.pipeline_thread.start()
        
        
        
        self.focal_length_x = None
    
    
    def depth_callback(self,msg: Float32):
        self.median_depth = msg.data
        if self.focal_length_x is not None:
            self.pixel_to_meter = self.median_depth/self.focal_length_x
    
    
    def run_pipeline_loop(self):
        # configure & start camera
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.pipeline.start(cfg)
        self.get_logger().info(f'RealSense started: {self.width}x{self.height}@{self.fps}fps')
        
        # grab intrinsics
        prof  = self.pipeline.get_active_profile()
        intr  = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.focal_length_x = intr.fx
        
        # sync device→ROS clock
        f0      = self.pipeline.wait_for_frames().get_color_frame()
        rs_t0   = f0.get_timestamp()/1000.0
        ros_now_msg = self.get_clock().now().to_msg()
        now_ros     = ros_now_msg.sec + ros_now_msg.nanosec * 1e-9
        self.device_to_ros_offset = now_ros - rs_t0
        self.prev_rs_t = rs_t0
        
        try:
            while rclpy.ok() and not self.shutdown_event.is_set():
                frames = self.pipeline.wait_for_frames()
                cf     = frames.get_color_frame()
                if not cf:
                    continue
                
                # **capture timestamp exactly here**
                rs_t        = cf.get_timestamp()/1000.0
                img         = np.asanyarray(cf.get_data())
                
                # enqueue for inference (drops if full)
                try:
                    self.frame_queue.put((img,rs_t),block=False)
                except Full:
                    pass
        finally:
            try:
                self.pipeline.stop()
            except Exception:
                pass
    
    
    def _inference_worker(self):
        """Runs LiteFlowNet3 on queued frames and stamps at capture time."""
        while rclpy.ok() and not self.shutdown_event.is_set():
            try:
                img, rs_t = self.frame_queue.get(timeout=0.1)
            except Exception:
                continue
            
            if self.shutdown_event.is_set():
                break
            
            # convert RS→ROS time
            ros_t     = rs_t + self.device_to_ros_offset
            stamp     = Time(sec=int(ros_t), nanosec=int((ros_t%1)*1e9))
            
            # preprocess
            rgb   = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            curr  = torch.from_numpy(rgb).permute(2,0,1).float()/255.0
            
            # skip first
            if self.prev_tensor is None:
                self.prev_tensor = curr
                self.prev_rs_t   = rs_t
                self.frame_queue.task_done()
                continue
            
            dt = rs_t - self.prev_rs_t
            dt = dt if dt>0 else 1e-3
            self.prev_rs_t = rs_t
            
            # inference + smoothing
            ten1 = (self.prev_tensor.to(self.device)-self.mean_one)
            ten2 = (curr.to(self.device)-self.mean_two)
            w32  = math.ceil(self.width/32)*32
            h32  = math.ceil(self.height/32)*32
            
            with torch.no_grad():
                i1   = torch.nn.functional.interpolate(ten1.unsqueeze(0),size=(h32,w32),mode='bilinear',align_corners=False)
                i2   = torch.nn.functional.interpolate(ten2.unsqueeze(0),size=(h32,w32),mode='bilinear',align_corners=False)
                flow = self.net(i1,i2)
                flow = torch.nn.functional.interpolate(flow,size=(self.height,self.width),mode='bilinear',align_corners=False)
                flow[:,0,:,:] *= self.width / w32
                flow[:,1,:,:] *= self.height/ h32
            
            arr   = flow[0].cpu().numpy()
            u_avg = np.mean(arr[0]) / dt
            vx    = float(u_avg * self.pixel_to_meter)
            self.velocity_buffer.append(vx)
            smooth= float(np.mean(self.velocity_buffer))
            
            # publish raw
            m_raw = Vector3Stamped()
            m_raw.header.stamp    = stamp
            m_raw.header.frame_id = 'camera_link'
            m_raw.vector.x        = vx
            self.flow_pub.publish(m_raw)
            
            # publish smooth
            m_smo = Vector3Stamped()
            m_smo.header.stamp    = stamp
            m_smo.header.frame_id = 'camera_link'
            m_smo.vector.x        = smooth
            self.smooth_pub.publish(m_smo)
            
            self.prev_tensor = curr
            self.frame_queue.task_done()
    
    
def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowDirectNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.get_logger().info('Shutting down optical_flow_direct_node…')
        # tell threads to bail out
        node.shutdown_event.set()
        # in case run_pipeline_loop is blocking in wait_for_frames
        try:
            node.pipeline.stop()
        except Exception:
            pass
        # wait up to a second for them to finish cleanly
        node.pipeline_thread.join(timeout=1.0)
        node.inference_thread.join(timeout=1.0)
        node.destroy_node()
        executor.shutdown() 
        rclpy.shutdown()



if __name__=='__main__':
    main()







