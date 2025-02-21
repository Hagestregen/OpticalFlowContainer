#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import torch
import numpy as np
import math
from .liteflownet import Network  # Import the LiteFlowNet model

class OpticalFlowNode(Node):
    def __init__(self):
        super().__init__('optical_flow_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        self.flow_pub = self.create_publisher(Image, 'optical_flow', 10)
        self.prev_tensor = None
        self.prev_time = None
        self.net = Network().cuda().eval()
        model_url = 'http://content.sniklaus.com/github/pytorch-liteflownet/network-default.pytorch'
        self.net.load_state_dict({
            k.replace('module', 'net'): v 
            for k, v in torch.hub.load_state_dict_from_url(model_url, progress=True).items()
        })
        self.mean_one = torch.tensor([0.411618, 0.434631, 0.454253]).view(3, 1, 1)
        self.mean_two = torch.tensor([0.410782, 0.433645, 0.452793]).view(3, 1, 1)
        self.frame_rate = 28.0  # Hz, for reference
        self.pixel_to_meter = 0.002428  # Adjusted meters per pixel
        self.get_logger().info('Initialized LiteFlowNet model')

    def image_callback(self, msg):
        try:
            img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            tensor = torch.from_numpy(img_np)
            if msg.encoding == 'rgb8':
                tensor = tensor[..., [2, 1, 0]]  # RGB to BGR
            elif msg.encoding != 'bgr8':
                self.get_logger().error(f'Unsupported image encoding: {msg.encoding}')
                return
            current_tensor = tensor.permute(2, 0, 1).float() / 255.0
        except Exception as e:
            self.get_logger().error(f'Failed to convert image to tensor: {str(e)}')
            return
        
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.prev_time is None:
            self.prev_tensor = current_tensor
            self.prev_time = current_time
            return
        
        dt = current_time - self.prev_time
        if dt <= 0:
            dt = 1e-3
        self.prev_time = current_time
        
        try:
            tenOne = (self.prev_tensor - self.mean_one).cuda()
            tenTwo = (current_tensor - self.mean_two).cuda()
            intWidth = msg.width
            intHeight = msg.height
            intPreprocessedWidth = int(math.ceil(math.ceil(intWidth / 32.0) * 32.0))
            intPreprocessedHeight = int(math.ceil(math.ceil(intHeight / 32.0) * 32.0))
            
            with torch.no_grad():
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
                tenFlow = self.net(tenPreprocessedOne, tenPreprocessedTwo) # / 20.0
                tenFlow = torch.nn.functional.interpolate(
                    input=tenFlow,
                    size=(intHeight, intWidth),
                    mode='bilinear',
                    align_corners=False
                )
                tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
                tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
            
            flow_np = tenFlow[0].cpu().numpy()  # [2, H, W]
            u_avg = np.mean(flow_np[0]) / dt  # pixels/second
            vx_m_per_s = u_avg * self.pixel_to_meter  # m/s
            
            flow_msg = Image()
            flow_msg.header = msg.header
            flow_msg.height = 1
            flow_msg.width = 1
            flow_msg.encoding = '32FC1'
            flow_msg.step = 4
            flow_msg.data = np.array([vx_m_per_s], dtype=np.float32).tobytes()
            
            self.flow_pub.publish(flow_msg)
            self.get_logger().info(f'Published camera velocity: vx={vx_m_per_s:.6f} m/s')
            
            self.prev_tensor = current_tensor
            
        except Exception as e:
            self.get_logger().error(f'Error computing optical flow: {str(e)}')
            return

def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down optical_flow_node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()