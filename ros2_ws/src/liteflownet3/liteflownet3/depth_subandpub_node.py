import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image, Range


class DepthCalculationNode(Node):
    def __init__(self):
        super().__init__('depth_calculation_node')
        
        # Declare parameter for depth_scale (default to 0.001, typical for D435i)
        self.declare_parameter('depth_scale', 0.001)
        self.depth_scale = self.get_parameter('depth_scale').get_parameter_value().double_value
        
        # Declare parameter for depth_mode (default to 'ROI')
        self.declare_parameter('depth_mode', 'ROI')
        
        # Subscriber to depth image topic from RealSense
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )
        
        # Publisher for median depth with timestamp
        self.publisher = self.create_publisher(Range, '/camera/depth/median_distance', 10)
        
        # Timer to publish at 1 Hz (same as original)
        self.timer = self.create_timer(0.1, self.calculate_and_publish)
        
        # Store latest depth image
        self.latest_depth_image = None
        
    def depth_callback(self, msg):
        """Store the latest depth image received from the topic."""
        self.latest_depth_image = msg
        
    def calculate_and_publish(self):
        """Calculate median depth from the latest image and publish it based on depth_mode."""
        if self.latest_depth_image is None:
            self.get_logger().warn("No depth image received yet")
            return
        
        # Convert depth image to numpy array
        depth_image = np.frombuffer(self.latest_depth_image.data, dtype=np.uint16).reshape(
            self.latest_depth_image.height, self.latest_depth_image.width
        )
        
        # Get the depth_mode parameter
        depth_mode = self.get_parameter('depth_mode').get_parameter_value().string_value
        if depth_mode not in ['ROI', 'entire']:
            self.get_logger().warn(f"Invalid depth_mode '{depth_mode}', defaulting to 'ROI'")
            depth_mode = 'ROI'
        
        if depth_mode == 'ROI':
            # Define central ROI in pixels
            h, w = depth_image.shape
            roi = depth_image[h//2-125:h//2+125, w//2-125:w//2+125]
            valid_depths = roi[roi > 0]
        else:  # depth_mode == 'entire'
            valid_depths = depth_image[depth_image > 0]
        
        if len(valid_depths) == 0:
            self.get_logger().warn(f"No valid depth data in {depth_mode}")
            return
        median_depth = np.median(valid_depths) * self.depth_scale
        
        # Create Range message with timestamp and distance
        # range_msg = Range()
        # range_msg.header.stamp = self.latest_depth_image.header.stamp  # Use depth image timestamp
        # range_msg.header.frame_id = self.latest_depth_image.header.frame_id  # Use depth image frame
        # range_msg.radiation_type = Range.INFRARED  # Suitable for depth camera
        # range_msg.field_of_view = 0.1  # Approximate for ROI or entire image
        # range_msg.min_range = 0.1  # D435i typical min range
        # range_msg.max_range = 10.0  # D435i typical max range
        # range_msg.range = float(median_depth)
        
        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.range        = float(median_depth)
        msg.field_of_view = 0.1  # approx
        msg.min_range     = 0.1
        msg.max_range     = 10.0
        self.publisher.publish(msg)
        
        # Publish the message
        # self.publisher.publish(range_msg)
        self.get_logger().info(f"Published median depth ({depth_mode}): {median_depth} meters")

def main(args=None):
    rclpy.init(args=args)
    node = DepthCalculationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down depth_calculation_node')
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()