#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import math
import time
import threading
from collections import deque

from .liteflownet import Network  # Adjust import based on your project structure

class OpticalFlowDirectNode(Node):
    def __init__(self):
        super().__init__('optical_flow_direct_node')
        
        # Declare parameters for resolution, frame rate, and pixel-to-meter conversion factor
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        # self.declare_parameter('pixel_to_meter', 0.000566) # 1280x720
        self.declare_parameter('pixel_to_meter', 0.000857) #640x480
        
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        self.pixel_to_meter = self.get_parameter('pixel_to_meter').value
        
        self.flow_pub = self.create_publisher(Vector3Stamped, '/optical_flow/LFN_velocity', 10)
        
        # Initialize LiteFlowNet
        self.get_logger().info('Initializing LiteFlowNet model...')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Network().cuda().eval()
        model_url = 'http://content.sniklaus.com/github/pytorch-liteflownet/network-default.pytorch'
        # Use the old state_dict format as before
        self.net.load_state_dict({
            k.replace('module', 'net'): v 
            for k, v in torch.hub.load_state_dict_from_url(model_url, progress=True).items()
        })
        self.get_logger().info('LiteFlowNet model loaded.')
        
        # Normalization tensors (adjust based on your training/expected inputs)
        self.mean_one = torch.tensor([0.411618, 0.434631, 0.454253]).view(3, 1, 1).to(self.device)
        self.mean_two = torch.tensor([0.410782, 0.433645, 0.452793]).view(3, 1, 1).to(self.device)
        
        # Variables to store the previous frame and its timestamp
        self.prev_tensor = None
        self.prev_time = None
        
        # Start the RealSense pipeline loop in a separate thread
        self.pipeline_thread = threading.Thread(target=self.run_pipeline_loop)
        self.pipeline_thread.daemon = True
        self.pipeline_thread.start()
    
    def run_pipeline_loop(self):
        # Configure and start the RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        pipeline.start(config)
        self.get_logger().info(f'RealSense pipeline started with resolution {self.width}x{self.height} at {self.fps} FPS.')
        
        try:
            while rclpy.ok():
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                
                # Convert the color image to a tensor of shape [3, H, W] and normalize to [0,1]
                current_tensor = torch.from_numpy(color_image).permute(2, 0, 1).float() / 255.0
                current_time = frames.get_timestamp() / 1000.0
                if self.prev_time is None:
                    self.prev_tensor = current_tensor
                    self.prev_time = current_time
                    continue
                dt = current_time - self.prev_time
                
                if self.prev_time is None:
                    self.prev_tensor = current_tensor
                    self.prev_time = current_time
                    continue
                
                dt = current_time - self.prev_time
                if dt <= 0:
                    dt = 1e-3
                self.prev_time = current_time
                
                try:
                    # Move CPU tensors to the correct device before subtracting the mean
                    tenOne = (self.prev_tensor.to(self.device) - self.mean_one)
                    tenTwo = (current_tensor.to(self.device) - self.mean_two)
                    
                    intWidth = self.width
                    intHeight = self.height
                    intPreprocessedWidth = int(math.ceil(math.ceil(intWidth / 32.0) * 32.0))
                    intPreprocessedHeight = int(math.ceil(math.ceil(intHeight / 32.0) * 32.0))
                    
                    with torch.no_grad():
                        # Resize inputs so their dimensions are multiples of 32
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
                        # Resize flow to original resolution
                        tenFlow = torch.nn.functional.interpolate(
                            input=tenFlow,
                            size=(intHeight, intWidth),
                            mode='bilinear',
                            align_corners=False
                        )
                        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
                        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
                    
                    # Convert flow to a numpy array and compute average horizontal flow
                    flow_np = tenFlow[0].cpu().numpy()  # Shape: [2, H, W]
                    u_avg = np.mean(flow_np[0]) / dt  # Horizontal flow (pixels/second)
                    vx_m_per_s = float(u_avg * self.pixel_to_meter)  # Convert to m/s
                    
                    
                    
                    
                    # Create and publish the velocity message
                    vel_msg = Vector3Stamped()
                    vel_msg.header.stamp = self.get_clock().now().to_msg()
                    vel_msg.header.frame_id = "camera_link"  # Adjust frame id as needed
                    vel_msg.vector.x = vx_m_per_s
                    vel_msg.vector.y = 0.0
                    vel_msg.vector.z = 0.0  # Assuming motion is primarily horizontal
                    
                    self.velocity_buffer = deque(maxlen=5)  # Store last 5 values to avoid noise
                    self.velocity_buffer.append(vx_m_per_s)
                    smoothed_vx = float(np.mean(self.velocity_buffer))
                    vel_msg.vector.x = smoothed_vx
                    
                    self.flow_pub.publish(vel_msg)
                    # self.get_logger().info(f'Published velocity: {vx_m_per_s:.6f} m/s')
                    
                    self.prev_tensor = current_tensor
                    
                    # Optional: display the image (press 'q' to exit display)
                    # cv2.imshow('Realsense Color Stream', color_image)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                
                except Exception as e:
                    self.get_logger().error(f'Error computing optical flow: {str(e)}')
        except Exception as e:
            self.get_logger().error(f'Error in pipeline loop: {str(e)}')
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

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
