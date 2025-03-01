#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
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
        
        self.image_width = 768
        self.image_height = 432
        self.device = torch.device('cuda')
        
        # self.model = NeuFlow().to(self.device)
        self.model = NeuFlow.from_pretrained("Study-is-happy/neuflow-v2").to(self.device)
        
        # checkpoint_path = os.path.join(
        #     get_package_share_directory('nueflow'),
        #     'NeuFlow_v2_master',
        #     'neuflow_mixed.pth'
        # )
        # self.get_logger().info(f"Loading checkpoint from: {checkpoint_path}")
        # if not os.path.exists(checkpoint_path):
        #     raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        # checkpoint = torch.load(checkpoint_path, map_location='cuda')
        # self.model.load_state_dict(checkpoint['model'], strict=True)
        
        self.vis_path = 'my_results_nohalf/'
        
        for m in self.model.modules():
            if type(m) is ConvBlock:
                m.conv1 = self.fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
                m.conv2 = self.fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
                delattr(m, "norm1")  # remove batchnorm
                delattr(m, "norm2")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        
        self.model.eval()
        self.model.half()
        self.model.init_bhwd(1, self.image_height, self.image_width, 'cuda')
        
        self.prev_image = None
        self.prev_time = None

    def fuse_conv_and_bn(self,conv, bn):
        """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
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

        # Prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # Prepare spatial bias
        b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv

    def get_cuda_image(self, img_np):
        """Resize image to 768x432 and prepare for GPU processing."""
        img_resized = cv2.resize(img_np, (self.image_width, self.image_height))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).half().to(self.device)
        # img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).to(self.device)
        return img_tensor[None]

    def image_callback(self, msg):
        """Process incoming ROS image messages to compute and visualize optical flow."""
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
                flow_np = flow.cpu().numpy()  # Shape: (2, H, W), 768x432 resolution
                
                # Handle NaN/Inf in flow
                flow_np = np.nan_to_num(flow_np, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Log raw flow for debugging (optional)
                u_raw = flow_np[0]
                v_raw = flow_np[1]
                u_avg_raw = np.mean(u_raw)
                u_std_raw = np.std(u_raw) if np.isfinite(u_raw).all() else 0.0
                self.get_logger().info(
                    f"Raw Flow - u min: {u_raw.min():.6f}, u max: {u_raw.max():.6f}, "
                    f"u_avg: {u_avg_raw:.6f}, u_std: {u_std_raw:.6f}, "
                    f"v min: {v_raw.min():.6f}, v max: {v_raw.max():.6f}"
                )
                
                # Save visualization every 0.2 seconds
                if int(current_time) % 0.5 == 0: # Save if time difference is at least 0.2 seconds
                    file_name = f"flow_{int(current_time * 1000)}.png"  # Use milliseconds for unique filenames
                    self.visualize_flow(flow_np, current_image, os.path.join(self.vis_path, file_name))
            
            self.prev_image = current_image
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
            
    # def visualize_flow(self, flow_np, image_tensor, filename="flow.png"):
    #     """Visualize optical flow and stack it with the input image, saving to disk."""
    #     if not os.path.exists(self.vis_path):
    #         os.makedirs(self.vis_path)
        
    #     # Convert flow to (H, W, 2) as expected by flow_viz.flow_to_image
    #     flow_permuted = np.transpose(flow_np, (1, 2, 0))  # (H, W, 2)
    #     flow_rgb = flow_viz.flow_to_image(flow_permuted.astype(np.float32))
        
    #     # Convert input image tensor to numpy, scale to [0, 255], and ensure uint8
    #     image_np = image_tensor[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3), float16
    #     image_np = (image_np * 255.0).astype(np.uint8)  # Scale to 0-255 and convert to uint8
        
    #     # Stack image and flow vertically
    #     combined = np.vstack([image_np, flow_rgb])
    #     cv2.imwrite(filename, combined)
    #     self.get_logger().info(f"Flow visualization saved to {filename}")
    def visualize_flow(self, flow_np, image_tensor, filename="flow.png"):
        """Visualize optical flow with vectors and stack it with the input image, saving to disk."""
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)
        
        # Convert flow to (H, W, 2) for visualization
        flow_permuted = np.transpose(flow_np, (1, 2, 0))  # (H, W, 2)
        flow_rgb = flow_viz.flow_to_image(flow_permuted.astype(np.float32))
        
        # Convert input image tensor to numpy, scale to [0, 255], and ensure uint8
        image_np = image_tensor[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3), float16
        image_np = (image_np * 255.0).astype(np.uint8)  # Scale to 0-255 and convert to uint8
        
        # Define 3x3 grid sections (768x432 -> 256x144 per section)
        h, w = flow_np.shape[1], flow_np.shape[2]  # 432, 768
        section_h, section_w = h // 3, w // 3  # 144, 256
        
        # Overlay vectors on the original image
        vector_image = image_np.copy()
        for i in range(3):
            for j in range(3):
                # Define section boundaries
                y_start, y_end = i * section_h, (i + 1) * section_h
                x_start, x_end = j * section_w, (j + 1) * section_w
                
                # Extract flow for this section
                section_flow = flow_np[:, y_start:y_end, x_start:x_end]
                u_mean = np.mean(section_flow[0])  # Horizontal flow
                v_mean = np.mean(section_flow[1])  # Vertical flow
                
                # Center point of the section
                center_x = x_start + section_w // 2
                center_y = y_start + section_h // 2
                
                # End point of the vector (scale for visibility)
                scale = 2  # Adjust this to make vectors more/less pronounced
                end_x = int(center_x + u_mean * scale)
                end_y = int(center_y + v_mean * scale)
                
                # Draw arrow on the image
                cv2.arrowedLine(
                    vector_image,
                    (center_x, center_y),  # Start point
                    (end_x, end_y),        # End point
                    (0, 255, 0),           # Green color
                    thickness=2,           # Line thickness
                    tipLength=0.3          # Arrowhead size
                )
        
        # Stack original image with vectors and flow visualization vertically
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