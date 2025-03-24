import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
from std_msgs.msg import Float32
import numpy as np

class DepthCalculationNode(Node):
    def __init__(self):
        super().__init__('depth_calculation_node')
        
        # Initialize RealSense pipeline for depth only
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        
        # Get depth scale
        profile = self.pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # Publisher for median depth
        self.publisher = self.create_publisher(Float32, '/camera/depth/median_distance', 10)
        
        # Timer to publish at 1 Hz
        self.timer = self.create_timer(1.0, self.calculate_and_publish)
        
    def calculate_and_publish(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return
        
        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Define central ROI (100x100 pixels)
        h, w = depth_image.shape
        roi = depth_image[h//2-50:h//2+50, w//2-50:w//2+50]
        
        # Calculate median depth in meters
        valid_depths = roi[roi > 0]  # Filter out invalid zeros
        if len(valid_depths) == 0:
            self.get_logger().warn("No valid depth data in ROI")
            return
        median_depth = np.median(valid_depths) * self.depth_scale
        
        # Publish median depth
        msg = Float32()
        msg.data = float(median_depth)
        self.publisher.publish(msg)
        self.get_logger().info(f"Published median depth: {median_depth} meters")

def main(args=None):
    rclpy.init(args=args)
    node = DepthCalculationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down depth_calculation_node')
    finally:
        node.pipeline.stop()
        rclpy.shutdown()

if __name__ == '__main__':
    main()



# import rclpy
# from rclpy.node import Node
# import pyrealsense2 as rs
# from std_msgs.msg import Float32
# import numpy as np

# class DepthCalculationNode(Node):
#     def __init__(self):
#         super().__init__('depth_calculation_node')
        
#         # Initialize RealSense pipeline
#         self.pipeline = rs.pipeline()
#         config = rs.config()
#         config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#         config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#         self.pipeline.start(config)
        
#         # Get camera intrinsics (focal length)
#         profile = self.pipeline.get_active_profile()
#         depth_sensor = profile.get_device().first_depth_sensor()
#         self.depth_scale = depth_sensor.get_depth_scale()
        
#         # Get color stream intrinsics
#         color_stream = profile.get_stream(rs.stream.color)
#         color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
#         self.focal_length_x = color_intrinsics.fx
#         self.get_logger().info(f"Color stream focal length (fx): {self.focal_length_x} pixels")
        
#         # Publisher for conversion factor
#         self.publisher = self.create_publisher(Float32, '/optical_flow/conversion_factor', 10)
        
#         # Timer to periodically publish conversion factor
#         self.timer = self.create_timer(1.0, self.calculate_and_publish)  # Update at 1 Hz
        
#     def calculate_and_publish(self):
#         frames = self.pipeline.wait_for_frames()
#         depth_frame = frames.get_depth_frame()
#         if not depth_frame:
#             return
        
#         # Convert depth frame to numpy array
#         depth_image = np.asanyarray(depth_frame.get_data())
        
#         # Define a central ROI (e.g., 100x100 pixels)
#         h, w = depth_image.shape
#         roi = depth_image[h//2-50:h//2+50, w//2-50:w//2+50]
        
#         # Calculate median depth in meters
#         valid_depths = roi[roi > 0]  # Filter out invalid zeros
#         if len(valid_depths) == 0:
#             self.get_logger().warn("No valid depth data in ROI")
#             return
#         median_depth = np.median(valid_depths) * self.depth_scale
#         self.get_logger().info(f"Median depth: {median_depth} meters")
        
#         # Calculate conversion factor (meters per pixel)
#         conversion_factor = median_depth / self.focal_length_x
        
#         # Publish the conversion factor
#         msg = Float32()
#         msg.data = float(conversion_factor)
#         self.publisher.publish(msg)
#         self.get_logger().info(f"Published conversion factor: {conversion_factor}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = DepthCalculationNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info('Shutting down depth_calculation_node')
#     finally:
#         node.pipeline.stop()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()