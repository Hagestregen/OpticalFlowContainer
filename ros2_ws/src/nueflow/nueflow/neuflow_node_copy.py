#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
import torch
import numpy as np
import cv2
import os
from .NeuFlow_v2_master.NeuFlow.neuflow import NeuFlow
from .NeuFlow_v2_master.NeuFlow.backbone_v7 import ConvBlock
from .NeuFlow_v2_master.data_utils import flow_viz  
from ament_index_python.packages import get_package_share_directory

class OpticalFlowNode(Node):
    def __init__(self):
        super().__init__('neuflow_node')
        self.subscription = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.velocity_pub = self.create_publisher(Float64, '/optical_flow/neuflow_velocity', 10)
        
        self.image_width = 768
        self.image_height = 432
        self.device = torch.device('cuda')
        self.pixel_to_meter = 0.000566
        
        self.model = NeuFlow().to(self.device)
        checkpoint_path = os.path.join(
            get_package_share_directory('nueflow'),
            'NeuFlow_v2_master',
            'neuflow_mixed.pth'
        )
        self.get_logger().info(f"Loading checkpoint from: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        self.model.load_state_dict(checkpoint['model'], strict=True)
        
        for m in self.model.modules():
            if isinstance(m, ConvBlock):
                m.conv1 = self.fuse_conv_and_bn(m.conv1, m.norm1)
                m.conv2 = self.fuse_conv_and_bn(m.conv2, m.norm2)
                delattr(m, "norm1")
                delattr(m, "norm2")
                m.forward = m.forward_fuse
        
        self.model.eval()
        # Force the model (and all submodules) to use float32.
        self.model.apply(lambda m: m.float())
        self.model.init_bhwd(1, self.image_height, self.image_width, 'cuda')
        
        self.prev_image = None
        self.prev_time = None
        self.get_logger().info('Initialized NeuFlow optical flow node')

    def fuse_conv_and_bn(self, conv, bn):
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
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
        b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
        return fusedconv

    def get_cuda_image(self, img_np):
        img_resized = cv2.resize(img_np, (self.image_width, self.image_height))
        # Convert image to float32 tensor (no half conversion)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().to(self.device)
        return img_tensor.unsqueeze(0)

    def image_callback(self, msg):
        try:
            img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            self.get_logger().info(f"Image dimensions: {msg.width}x{msg.height}")
            if msg.encoding == 'rgb8':
                img_np = img_np[..., [2, 1, 0]]  # Convert RGB to BGR
            elif msg.encoding != 'bgr8':
                self.get_logger().error(f'Unsupported image encoding: {msg.encoding}')
                return
            
            current_image = self.get_cuda_image(img_np)
            current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            if self.prev_image is None:
                self.prev_image = current_image
                self.prev_time = current_time
                return
            
            dt = current_time - self.prev_time
            if dt <= 0:
                dt = 1e-3
            self.prev_time = current_time
            
            with torch.no_grad():
                flow = self.model(self.prev_image, current_image)[-1][0]
                flow_np = flow.cpu().numpy()  # Shape: (2, H, W)
                
                # Handle NaN/Inf in flow.
                u_raw = np.nan_to_num(flow_np[0], nan=0.0, posinf=0.0, neginf=0.0)
                v_raw = np.nan_to_num(flow_np[1], nan=0.0, posinf=0.0, neginf=0.0)
                
                u_avg = np.mean(u_raw)
                vx_m_per_s = (u_avg / dt) * self.pixel_to_meter
                
                self.get_logger().info(
                    f"Scaled Flow - u min: {flow_np[0].min():.6f}, u max: {flow_np[0].max():.6f}, "
                    f"u_avg: {u_avg:.6f}, dt: {dt:.6f}, vx: {vx_m_per_s:.6f}"
                )
                
                # Visualize flow every 2 seconds.
                if int(current_time) % 2 == 0:
                    self.visualize_flow(flow_np, current_image, f"flow_{int(current_time)}.png")
            
            velocity_msg = Float64()
            velocity_msg.data = float(vx_m_per_s)
            self.velocity_pub.publish(velocity_msg)
            
            self.prev_image = current_image
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
            
    def visualize_flow(self, flow_np, image_tensor, filename="flow.png"):
        flow_permuted = np.transpose(flow_np, (1, 2, 0))  # (H, W, 2)
        flow_rgb = flow_viz.flow_to_image(flow_permuted.astype(np.float32))
        
        image_np = image_tensor[0].cpu().numpy().transpose(1, 2, 0)
        image_np = image_np.astype(np.uint8)
        
        combined = np.vstack([image_np, flow_rgb])
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



# class OpticalFlowNode(Node):
#     def __init__(self):
#         """
#         Initialize the NeuFlow optical flow node.
#         """
#         super().__init__('neuflow_node')
#         self.subscription = self.create_subscription(
#             Image, '/camera/camera/color/image_raw', self.image_callback, 10)
#         self.velocity_pub = self.create_publisher(Float64, '/optical_flow/neuflow_velocity', 10)
        
#         self.image_width = 768
#         self.image_height = 432
#         self.device = torch.device('cuda')
#         self.pixel_to_meter = 0.000566
        
#         self.model = NeuFlow().to(self.device)
#         checkpoint_path = os.path.join(
#             get_package_share_directory('nueflow'),
#             'NeuFlow_v2_master',
#             'neuflow_mixed.pth'
#         )
#         self.get_logger().info(f"Loading checkpoint from: {checkpoint_path}")
#         if not os.path.exists(checkpoint_path):
#             raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
#         checkpoint = torch.load(checkpoint_path, map_location='cuda')
#         self.model.load_state_dict(checkpoint['model'], strict=True)
        
#         for m in self.model.modules():
#             if isinstance(m, ConvBlock):
#                 m.conv1 = self.fuse_conv_and_bn(m.conv1, m.norm1)
#                 m.conv2 = self.fuse_conv_and_bn(m.conv2, m.norm2)
#                 delattr(m, "norm1")
#                 delattr(m, "norm2")
#                 m.forward = m.forward_fuse
        
#         self.model.eval()
#         self.model.half()
#         self.model.init_bhwd(1, self.image_height, self.image_width, 'cuda')
        
#         self.prev_image = None
#         self.prev_time = None
#         self.get_logger().info('Initialized NeuFlow optical flow node')

#     def fuse_conv_and_bn(self, conv, bn):
#         fusedconv = (
#             torch.nn.Conv2d(
#                 conv.in_channels,
#                 conv.out_channels,
#                 kernel_size=conv.kernel_size,
#                 stride=conv.stride,
#                 padding=conv.padding,
#                 dilation=conv.dilation,
#                 groups=conv.groups,
#                 bias=True,
#             )
#             .requires_grad_(False)
#             .to(conv.weight.device)
#         )
#         w_conv = conv.weight.clone().view(conv.out_channels, -1)
#         w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
#         fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
#         b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
#         b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
#         fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
#         return fusedconv

#     def get_cuda_image(self, img_np):
#         img_resized = cv2.resize(img_np, (self.image_width, self.image_height))
#         img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).half().to(self.device)
#         return img_tensor[None]

#     def image_callback(self, msg):
#         try:
#             img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
#             self.get_logger().info(f"Image dimensions: {msg.width}x{msg.height}")
#             if msg.encoding == 'rgb8':
#                 img_np = img_np[..., [2, 1, 0]]  # Convert RGB to BGR
#             elif msg.encoding != 'bgr8':
#                 self.get_logger().error(f'Unsupported image encoding: {msg.encoding}')
#                 return
            
#             current_image = self.get_cuda_image(img_np)
#             current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
#             if self.prev_image is None:
#                 self.prev_image = current_image
#                 self.prev_time = current_time
#                 return
            
#             dt = current_time - self.prev_time
#             if dt <= 0:
#                 dt = 1e-3
#             self.prev_time = current_time
            
#             with torch.no_grad():
#                 flow = self.model(self.prev_image, current_image)[-1][0]
#                 flow_np = flow.cpu().numpy()  # Shape: (2, H, W)
                
#                 # Handle NaN/Inf in flow
#                 u_raw = flow_np[0]
#                 v_raw = flow_np[1]
#                 u_raw = np.nan_to_num(u_raw, nan=0.0, posinf=0.0, neginf=0.0)
#                 v_raw = np.nan_to_num(v_raw, nan=0.0, posinf=0.0, neginf=0.0)
                
#                 # Log raw flow
#                 u_avg_raw = np.mean(u_raw)
#                 u_std_raw = np.std(u_raw) if np.isfinite(u_raw).all() else 0.0  # Safe std calculation
#                 self.get_logger().info(
#                     f"Raw Flow - u min: {u_raw.min():.6f}, u max: {u_raw.max():.6f}, "
#                     f"u_avg: {u_avg_raw:.6f}, u_std: {u_std_raw:.6f}, "
#                     f"v min: {v_raw.min():.6f}, v max: {v_raw.max():.6f}"
#                 )
                
#                 # Apply scaling (test without it first by commenting out)
#                 # flow_np[0] *= (msg.width / self.image_width)
#                 # flow_np[1] *= (msg.height / self.image_height)
                
#                 u_avg = np.mean(flow_np[0])
#                 vx_m_per_s = (u_avg / dt) * self.pixel_to_meter
                
#                 self.get_logger().info(
#                     f"Scaled Flow - u min: {flow_np[0].min():.6f}, u max: {flow_np[0].max():.6f}, "
#                     f"u_avg: {u_avg:.6f}, dt: {dt:.6f}, vx: {vx_m_per_s:.6f}"
#                 )
                
#                 # Visualize every 5 seconds
#                 if int(current_time) % 2 == 0:
#                     self.visualize_flow(flow_np, current_image, f"flow_{int(current_time)}.png")
            
#             velocity_msg = Float64()
#             velocity_msg.data = float(vx_m_per_s)
#             self.velocity_pub.publish(velocity_msg)
            
#             self.prev_image = current_image
            
#         except Exception as e:
#             self.get_logger().error(f'Error processing image: {str(e)}')
            
            
#     def visualize_flow(self, flow_np, image_tensor, filename="flow.png"):
#         """
#         Visualize optical flow using flow_viz.flow_to_image, stacking it with the input image.
        
#         Args:
#             flow_np: Flow array (2, H, W) with x and y components
#             image_tensor: Input image tensor (1, 3, H, W) for stacking
#             filename: Output file name for the visualization
#         """
#         # Convert flow to (H, W, 2) as expected by flow_to_image
#         flow_permuted = np.transpose(flow_np, (1, 2, 0))  # From (2, H, W) to (H, W, 2)
#         flow_rgb = flow_viz.flow_to_image(flow_permuted)  # Returns uint8 array (H, W, 3)
        
#         # Convert input image tensor to numpy, scale to [0, 255], and ensure uint8
#         image_np = image_tensor[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3), float16
#         image_np = (image_np * 255.0).astype(np.uint8)  # Scale to 0-255 and convert to uint8
        
#         # Stack image and flow vertically
#         combined = np.vstack([image_np, flow_rgb])
#         cv2.imwrite(filename, combined)
#         self.get_logger().info(f"Flow visualization saved to {filename}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = OpticalFlowNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info('Shutting down neuflow_node')
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
    
    
