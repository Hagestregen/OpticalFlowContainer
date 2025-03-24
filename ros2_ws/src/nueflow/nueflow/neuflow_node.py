#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import threading
import os
from .NeuFlow_v2_master.NeuFlow.neuflow import NeuFlow
from .NeuFlow_v2_master.NeuFlow.backbone_v7 import ConvBlock
from .NeuFlow_v2_master.data_utils import flow_viz
from ament_index_python.packages import get_package_share_directory
import safetensors

class OpticalFlowNode(Node):
    def __init__(self):
        super().__init__('neuflow_node')
        
        # Declare parameters for camera configuration
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        
        # Retrieve parameter values
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        
        # NeuFlow-specific dimensions
        self.image_width = 768
        self.image_height = 432
        self.device = torch.device('cuda')
        
        # Initialize NeuFlow model
        self.get_logger().info('Initializing NeuFlow model...')
        self.model = NeuFlow.from_pretrained("Study-is-happy/neuflow-v2").to(self.device)
        
        # Optimize model by fusing Conv and BatchNorm layers
        for m in self.model.modules():
            if type(m) is ConvBlock:
                m.conv1 = self.fuse_conv_and_bn(m.conv1, m.norm1)  # Update conv1
                m.conv2 = self.fuse_conv_and_bn(m.conv2, m.norm2)  # Update conv2
                delattr(m, "norm1")  # Remove batchnorm1
                delattr(m, "norm2")  # Remove batchnorm2
                m.forward = m.forward_fuse  # Update forward method
        
        self.model.eval()
        self.model.half()  # Use half-precision for efficiency
        self.model.init_bhwd(1, self.image_height, self.image_width, 'cuda')
        self.get_logger().info('NeuFlow model loaded.')
        
        # Visualization path
        self.vis_path = 'my_results_nohalf/'
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)
        
        # Variables for previous frame and timing
        self.prev_image = None
        self.prev_time = None
        
        # Start RealSense pipeline in a separate thread
        self.pipeline_thread = threading.Thread(target=self.run_pipeline_loop)
        self.pipeline_thread.daemon = True
        self.pipeline_thread.start()

    def fuse_conv_and_bn(self, conv, bn):
        """Fuse Conv2d and BatchNorm2d layers for optimization."""
        fusedconv = (
            torch.nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=True,
            )
            .requires_grad_(False)
            .to(conv.weight.device)
        )
        # Fuse weights
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
        # Fuse biases
        b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
        return fusedconv

    def get_cuda_image(self, img_np):
        """Resize image to 768x432 and prepare for GPU processing."""
        img_resized = cv2.resize(img_np, (self.image_width, self.image_height))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).to(self.device).float() / 255.0
        img_tensor = img_tensor.half()  # Convert to half-precision
        return img_tensor[None]  # Add batch dimension

    def run_pipeline_loop(self):
        """Capture and process RealSense camera stream."""
        # Configure RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        pipeline.start(config)
        self.get_logger().info(f'RealSense pipeline started with {self.width}x{self.height} at {self.fps} FPS.')
        
        try:
            while rclpy.ok():
                # Wait for frames
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                
                # Convert BGR to RGB
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
                # Prepare current image tensor
                current_image = self.get_cuda_image(color_image)
                current_time = frames.get_timestamp() / 1000.0  # Convert ms to seconds
                
                # Skip first frame
                if self.prev_image is None:
                    self.prev_image = current_image
                    self.prev_time = current_time
                    continue
                
                # Compute time difference
                dt = current_time - self.prev_time
                if dt <= 0:
                    dt = 1e-3  # Prevent division by zero
                self.prev_time = current_time
                
                try:
                    with torch.no_grad():
                        # Compute optical flow
                        flow = self.model(self.prev_image, current_image)[-1][0]
                        flow_np = flow.cpu().numpy()  # Shape: (2, H, W), 768x432 resolution
                        
                        # Handle NaN/Inf values in flow
                        flow_np = np.nan_to_num(flow_np, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # Log flow statistics for debugging
                        u_raw = flow_np[0]
                        v_raw = flow_np[1]
                        u_avg_raw = np.mean(u_raw)
                        u_std_raw = np.std(u_raw) if np.isfinite(u_raw).all() else 0.0
                        self.get_logger().info(
                            f"Raw Flow - u min: {u_raw.min():.6f}, u max: {u_raw.max():.6f}, "
                            f"u_avg: {u_avg_raw:.6f}, u_std: {u_std_raw:.6f}, "
                            f"v min: {v_raw.min():.6f}, v max: {v_raw.max():.6f}"
                        )
                        
                        # Save visualization every 0.5 seconds
                        if int(current_time * 2) % 1 == 0:  # Check every 0.5 seconds
                            file_name = f"flow_{int(current_time * 1000)}.png"
                            self.visualize_flow(flow_np, current_image, os.path.join(self.vis_path, file_name))
                    
                    # Update previous image
                    self.prev_image = current_image
                
                except Exception as e:
                    self.get_logger().error(f'Error computing optical flow: {str(e)}')
        
        except Exception as e:
            self.get_logger().error(f'Error in pipeline loop: {str(e)}')
        finally:
            pipeline.stop()
            self.get_logger().info('RealSense pipeline stopped.')

    def visualize_flow(self, flow_np, image_tensor, filename="flow.png"):
        """Visualize optical flow with vectors and save to disk."""
        # Convert flow to (H, W, 2) for visualization
        flow_permuted = np.transpose(flow_np, (1, 2, 0))  # (H, W, 2)
        flow_rgb = flow_viz.flow_to_image(flow_permuted.astype(np.float32))
        
        # Convert input image tensor to numpy for visualization
        image_np = image_tensor[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3), float16
        image_np = (image_np * 255.0).astype(np.uint8)  # Scale to 0-255
        
        # Define 3x3 grid for vector overlay (768x432 -> 256x144 per section)
        h, w = flow_np.shape[1], flow_np.shape[2]  # 432, 768
        section_h, section_w = h // 3, w // 3  # 144, 256
        
        # Overlay vectors on the original image
        vector_image = image_np.copy()
        for i in range(3):
            for j in range(3):
                y_start, y_end = i * section_h, (i + 1) * section_h
                x_start, x_end = j * section_w, (j + 1) * section_w
                
                # Compute mean flow in section
                section_flow = flow_np[:, y_start:y_end, x_start:x_end]
                u_mean = np.mean(section_flow[0])  # Horizontal flow
                v_mean = np.mean(section_flow[1])  # Vertical flow
                
                # Center of the section
                center_x = x_start + section_w // 2
                center_y = y_start + section_h // 2
                
                # End point of vector (scaled for visibility)
                scale = 2
                end_x = int(center_x + u_mean * scale)
                end_y = int(center_y + v_mean * scale)
                
                # Draw arrow
                cv2.arrowedLine(
                    vector_image,
                    (center_x, center_y),
                    (end_x, end_y),
                    (0, 255, 0),  # Green
                    thickness=2,
                    tipLength=0.3
                )
        
        # Stack images vertically
        combined = np.vstack([vector_image, flow_rgb])
        cv2.imwrite(filename, combined)
        self.get_logger().info(f"Flow visualization saved to {filename}")

def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down neuflow_node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()