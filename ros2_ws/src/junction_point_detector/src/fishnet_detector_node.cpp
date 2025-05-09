#include "junction_point_detector/fishnet_detector_ros.h"

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FishnetDetectorNode>());
    rclcpp::shutdown();
    return 0;
}