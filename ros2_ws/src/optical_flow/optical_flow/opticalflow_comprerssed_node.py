#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import torch
import numpy as np
import math
from .liteflownet import Network  # Import the LiteFlowNet model
from std_msgs.msg import Float64
import time
import cv2

class OpticalFlowNode(Node):
    def __init__(self):
        super().__init__('optical_flow_node')
        # Subscribe to the compressed image topic
        self.subscription = self.create_subscription(
            CompressedImage,  # Changed from Image to CompressedImage
            '/camera/camera/color/image_raw/compressed',  # Corrected topic name
            self.image_callback,
            10
        )
        # Publisher for optical flow velocity
        self.flow_pub = self.create_publisher(Float64, 'optical_flow', 10)
        self.prev_tensor = None
        self.prev_time = None
        # Initialize LiteFlowNet model
        self.net = Network().cuda().eval()
        model_url = 'http://content.sniklaus.com/github/pytorch-liteflownet/network-default.pytorch'
        self.net.load_state_dict({
            k.replace('module', 'net'): v 
            for k, v in torch.hub.load_state_dict_from_url(model_url, progress=True).items()
        })
        # Mean values for image preprocessing
        self.mean_one = torch.tensor([0.411618, 0.434631, 0.454253]).view(3, 1, 1)
        self.mean_two = torch.tensor([0.410782, 0.433645, 0.452793]).view(3, 1, 1)
        self.frame_rate = 28.0  # Hz, for reference
        self.pixel_to_meter = 0.000566  # Adjusted meters per pixel
        self.get_logger().info('Initialized LiteFlowNet model')

    def image_callback(self, msg):
        try:
            # Decode the compressed image (JPEG format)
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                self.get_logger().error("Failed to decode compressed image")
                return
            self.get_logger().info(f"Decoded image shape: {cv_image.shape}, encoding: {msg.format}")

            # Convert image to tensor (assuming BGR from JPEG)
            tensor = torch.from_numpy(cv_image)
            current_tensor = tensor.permute(2, 0, 1).float() / 255.0  # BGR to [C, H, W], normalize
        except Exception as e:
            self.get_logger().error(f'Failed to convert image to tensor: {str(e)}')
            return

        # Get current timestamp from the message header
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.prev_time is None:
            self.prev_tensor = current_tensor
            self.prev_time = current_time
            return

        # Calculate time difference
        dt = current_time - self.prev_time
        if dt <= 0:
            dt = 1e-3  # Avoid division by zero
        self.prev_time = current_time

        try:
            # Preprocess tensors for LiteFlowNet
            tenOne = (self.prev_tensor - self.mean_one).cuda()
            tenTwo = (current_tensor - self.mean_two).cuda()
            intWidth = cv_image.shape[1]  # Width from decoded image
            intHeight = cv_image.shape[0]  # Height from decoded image
            intPreprocessedWidth = int(math.ceil(math.ceil(intWidth / 32.0) * 32.0))
            intPreprocessedHeight = int(math.ceil(math.ceil(intHeight / 32.0) * 32.0))

            with torch.no_grad():
                # Resize tensors for LiteFlowNet
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
                tenFlow = torch.nn.functional.interpolate(
                    input=tenFlow,
                    size=(intHeight, intWidth),
                    mode='bilinear',
                    align_corners=False
                )
                # Adjust flow for original image size
                tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
                tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

            # Convert flow to numpy and compute average velocity
            flow_np = tenFlow[0].cpu().numpy()  # [2, H, W]
            u_avg = np.mean(flow_np[0]) / dt  # pixels/second
            vx_m_per_s = float(u_avg * self.pixel_to_meter)  # m/s

            # Publish velocity
            velocity_msg = Float64()
            velocity_msg.data = float(vx_m_per_s)
            self.flow_pub.publish(velocity_msg)
            self.get_logger().info(f'Published camera velocity: vx={vx_m_per_s:.6f} m/s, dt={dt:.6f}')

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