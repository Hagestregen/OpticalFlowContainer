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
        threading.Thread(target=self._inference_worker,daemon=True).start()
        threading.Thread(target=self.run_pipeline_loop, daemon=True).start()
        
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
            while rclpy.ok():
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
            self.pipeline.stop()
    
    
    def _inference_worker(self):
        """Runs LiteFlowNet3 on queued frames and stamps at capture time."""
        while rclpy.ok():
            img, rs_t = self.frame_queue.get()
            
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()







# import rclpy
# from rclpy.node import Node
# from builtin_interfaces.msg import Time
# from geometry_msgs.msg import Vector3Stamped
# from std_msgs.msg import Float32
# import pyrealsense2 as rs
# import numpy as np
# import cv2
# import torch
# import math
# import time
# import threading
# from collections import deque
# import os
# from ament_index_python.packages import get_package_share_directory  # Add this import
# import csv

# from .liteflownet3 import Network

# class OpticalFlowDirectNode(Node):
#     def __init__(self):
#         super().__init__('optical_flow_direct_node')
        
#         # Declare parameters
#         self.declare_parameter('width', 640)
#         self.declare_parameter('height', 480)
#         self.declare_parameter('fps', 30)
#         self.declare_parameter('pixel_to_meter', 0.000857)  # For 640x480
        
#         # Retrieve parameter values
#         self.width = self.get_parameter('width').value
#         self.height = self.get_parameter('height').value
#         self.fps = self.get_parameter('fps').value
#         self.pixel_to_meter = self.get_parameter('pixel_to_meter').value
        
#         self.writeCsv = False
        
#         if self.writeCsv:
#             # Prepare CSV file for inference times
#             self.csv_filename = f"lfn3_direct_inference_{self.width}x{self.height}.csv"
#             # Write header if new file
#             if not os.path.isfile(self.csv_filename):
#                 with open(self.csv_filename, 'w', newline='') as csvfile:
#                     writer = csv.writer(csvfile)
#                     writer.writerow(['timestamp', 'inference_time_s'])
        
        
        
#         # Publisher for velocity
#         self.flow_pub = self.create_publisher(Vector3Stamped, '/optical_flow/LFN3_velocity', 10)
#         self.smooth_flow_pub = self.create_publisher(Vector3Stamped, '/optical_flow/LFN3_smooth_velocity', 10)
        
#         # Initialize LiteFlowNet3 model
#         self.get_logger().info('Initializing LiteFlowNet3 model...')
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.net = Network().to(self.device).eval()
        
#         # Use get_package_share_directory to locate the model file
#         package_share_dir = get_package_share_directory('liteflownet3')
#         model_path = os.path.join(package_share_dir, 'network-sintel.pytorch')
        
#         # Load the model weights
#         self.net.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.get_logger().info('LiteFlowNet3 model loaded.')
        
#         # Normalization tensors (verify these match LiteFlowNet3 training)
#         self.mean_one = torch.tensor([0.411618, 0.434631, 0.454253]).view(3, 1, 1).to(self.device)
#         self.mean_two = torch.tensor([0.410782, 0.433645, 0.452793]).view(3, 1, 1).to(self.device)
        
#         # Variables for previous frame and smoothing
#         self.prev_tensor = None
#         # self.prev_time = None
#         self.velocity_buffer = deque(maxlen=3)
        
#         # Subscription to median depth
#         self.median_depth = None
#         self.sub_depth = self.create_subscription(
#             Float32,
#             '/camera/depth/median_distance',
#             self.depth_callback,
#             10
#         )
        
#         # Initialize the RealSense pipeline
#         self.pipeline = rs.pipeline()
        
#         # Start RealSense pipeline in a thread
#         self.pipeline_thread = threading.Thread(target=self.run_pipeline_loop)
#         self.pipeline_thread.daemon = True
#         self.pipeline_thread.start()
        
#         self.focal_length_x = None
        
#     def depth_callback(self, msg):
#         self.median_depth = msg.data
#         if self.focal_length_x is not None:
#             self.pixel_to_meter = self.median_depth / self.focal_length_x
#             # self.get_logger().info(f"Updated pixel_to_meter: {self.pixel_to_meter:.6f}")
    
#     def run_pipeline_loop(self):
#         # Configure RealSense pipeline
#         config = rs.config()
#         config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
#         self.pipeline.start(config)
#         self.get_logger().info(f'RealSense pipeline started with {self.width}x{self.height} at {self.fps} FPS.')
        
#         # Get focal length from color stream
#         profile = self.pipeline.get_active_profile()
#         color_stream = profile.get_stream(rs.stream.color)
#         intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
#         self.focal_length_x = intrinsics.fx
#         self.get_logger().info(f"Focal length: {self.focal_length_x} pixels")
        
#         frames0    = self.pipeline.wait_for_frames()
#         color_frame0 = frames0.get_color_frame()
#         rs_t0      = color_frame0.get_timestamp() / 1000.0        # device clock
#         ros_t0_msg = self.get_clock().now().to_msg()              # ROS clock
#         ros_t0     = ros_t0_msg.sec + ros_t0_msg.nanosec * 1e-9
#         self.device_to_ros_offset = ros_t0 - rs_t0      
#         self.prev_rs_t = rs_t0
#         self.get_logger().info(f"Device to ROS offset: {self.device_to_ros_offset:.6f} seconds")
    
        
#         # Configure camera settings: disable autoexposure, set exposure to 500, set gain to 10
#         # device = profile.get_device()
#         # sensors = device.query_sensors()
#         # for sensor in sensors:
#         #     if not sensor.is_depth_sensor():
#         #         color_sensor = sensor
#         #         break
#         # else:
#         #     self.get_logger().error("Color sensor not found")
#         #     return
        
#         # # Disable autoexposure
#         # if color_sensor.supports(rs.option.enable_auto_exposure):
#         #     color_sensor.set_option(rs.option.enable_auto_exposure, 0)
#         #     self.get_logger().info("Autoexposure disabled")
#         # else:
#         #     self.get_logger().warn("Autoexposure option not supported")
        
#         # # Set exposure to 500
#         # if color_sensor.supports(rs.option.exposure):
#         #     color_sensor.set_option(rs.option.exposure, 500)
#         #     self.get_logger().info("Exposure set to 500")
#         # else:
#         #     self.get_logger().warn("Exposure option not supported")
        
#         # # Set gain to 10
#         # if color_sensor.supports(rs.option.gain):
#         #     color_sensor.set_option(rs.option.gain, 10)
#         #     self.get_logger().info("Gain set to 10")
#         # else:
#         #     self.get_logger().warn("Gain option not supported")
        
#         try:
#             while rclpy.ok():
#                 frames = self.pipeline.wait_for_frames()
 
#                 color_frame = frames.get_color_frame()
#                 if not color_frame:
#                     continue
                
#                 rs_t = color_frame.get_timestamp() / 1000.0    # seconds on device clock
#                 ros_t = rs_t + self.device_to_ros_offset  
#                  # convert ros_t0 to float seconds

#                 ros_stamp = Time()
#                 sec  = int(ros_t)
#                 nsec = int((ros_t - sec) * 1e9)
#                 ros_stamp.sec = sec
#                 ros_stamp.nanosec = nsec
                
                
                
                
#                 # ros_stamp = self.get_clock().now().to_msg()
#                 color_image = np.asanyarray(color_frame.get_data())
                
#                 if self.writeCsv:
#                     start_time = time.perf_counter()

                
#                 # Convert BGR to RGB
#                 color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
#                 # Convert to tensor and normalize
#                 current_tensor = torch.from_numpy(color_image).permute(2, 0, 1).float() / 255.0
#                 # current_time = frames.get_timestamp() / 1000.0  # ms to seconds
                
                
#                 # Skip first frame
#                 if self.prev_tensor is None:
#                     self.prev_tensor = current_tensor
#                     # self.prev_time = current_time
#                     continue
                
#                 # Time difference
#                 dt        = rs_t - self.prev_rs_t
#                 if dt <= 0:
#                     dt = 1e-3  # Prevent division by zero
#                 self.prev_rs_t = rs_t
                
#                 try:
#                     # Preprocess tensors
#                     tenOne = (self.prev_tensor.to(self.device) - self.mean_one)
#                     tenTwo = (current_tensor.to(self.device) - self.mean_two)
                    
#                     # Compute dimensions for resizing
#                     intWidth = self.width
#                     intHeight = self.height
#                     intPreprocessedWidth = int(math.ceil(math.ceil(intWidth / 32.0) * 32.0))
#                     intPreprocessedHeight = int(math.ceil(math.ceil(intHeight / 32.0) * 32.0))
                    
#                     with torch.no_grad():
#                         # Resize inputs
#                         tenPreprocessedOne = torch.nn.functional.interpolate(
#                             input=tenOne.unsqueeze(0),
#                             size=(intPreprocessedHeight, intPreprocessedWidth),
#                             mode='bilinear',
#                             align_corners=False
#                         )
#                         tenPreprocessedTwo = torch.nn.functional.interpolate(
#                             input=tenTwo.unsqueeze(0),
#                             size=(intPreprocessedHeight, intPreprocessedWidth),
#                             mode='bilinear',
#                             align_corners=False
#                         )
#                         # Compute optical flow
#                         tenFlow = self.net(tenPreprocessedOne, tenPreprocessedTwo)
#                         # Resize flow back
#                         tenFlow = torch.nn.functional.interpolate(
#                             input=tenFlow,
#                             size=(intHeight, intWidth),
#                             mode='bilinear',
#                             align_corners=False
#                         )
#                         tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
#                         tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
                    
#                     # # Crop to upper two-thirds
#                     # cropped_height = int(intHeight * 2 / 3)
#                     # tenFlow_cropped = tenFlow[:, :, :cropped_height, :]
                                        
#                     # Compute velocity
#                     flow_np = tenFlow[0].cpu().numpy()  # [2, H, W]
#                     # flow_np = tenFlow_cropped[0].cpu().numpy()
#                     u_avg = np.mean(flow_np[0]) / dt  # Horizontal flow (pixels/s)
#                     vx_m_per_s = float(u_avg * self.pixel_to_meter)  # Meters/s
                    
#                     # Smooth velocity
#                     self.velocity_buffer.append(vx_m_per_s)
#                     smoothed_vx = float(np.mean(self.velocity_buffer))
                    
#                     # Publish velocity
#                     vel_msg = Vector3Stamped()
#                     vel_msg.header.stamp = ros_stamp
#                     # vel_msg.header.stamp.sec       = int(ros_t)
#                     # vel_msg.header.stamp.nanosec   = int((ros_t - int(ros_t)) * 1e9)
#                     vel_msg.header.frame_id = "camera_link"
#                     vel_msg.vector.x = vx_m_per_s
#                     vel_msg.vector.y = 0.0  # Add vertical flow if needed
#                     vel_msg.vector.z = 0.0
                    
#                     # Publish velocity
#                     vel_smooth_msg = Vector3Stamped()
#                     vel_smooth_msg.header.stamp = ros_stamp
#                     # vel_msg.header.stamp.sec       = int(ros_t)
#                     # vel_msg.header.stamp.nanosec   = int((ros_t - int(ros_t)) * 1e9)
#                     vel_smooth_msg.header.frame_id = "camera_link"
#                     vel_smooth_msg.vector.x = smoothed_vx
#                     vel_smooth_msg.vector.y = 0.0  # Add vertical flow if needed
#                     vel_smooth_msg.vector.z = 0.0
                    
#                     self.flow_pub.publish(vel_msg)
#                     # self.get_logger().info(f"flow is: {vel_msg.vector.x}")
#                     self.smooth_flow_pub.publish(vel_smooth_msg)
#                     # self.get_logger().info(f'Published velocity: {smoothed_vx:.6f} m/s')
                    
#                     # Update previous tensor
#                     self.prev_tensor = current_tensor
                    
#                     if self.writeCsv:
#                         end_time = time.perf_counter()
#                         inference_time = end_time - start_time    
                    
#                         # Append to CSV   
#                         timestamp = time.time()   
#                         with open(self.csv_filename, 'a', newline='') as csvfile: 
#                             csv.writer(csvfile).writerow([timestamp, inference_time]) 
                
#                 except Exception as e:
#                     self.get_logger().error(f'Error computing optical flow: {str(e)}')
        
#         except Exception as e:
#             self.get_logger().error(f'Error in pipeline loop: {str(e)}')
#         finally:
#             self.pipeline.stop()

# def main(args=None):
#     rclpy.init(args=args)
#     node = OpticalFlowDirectNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info('Shutting down optical_flow_direct_node')
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()



