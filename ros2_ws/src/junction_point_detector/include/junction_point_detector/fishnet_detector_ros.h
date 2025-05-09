#ifndef FISHNET_DETECTOR_ROS_H
#define FISHNET_DETECTOR_ROS_H

// Libraries for ROS
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "nav_msgs/msg/odometry.hpp"
// #include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/point32.hpp"
#include "sensor_msgs/msg/point_cloud.hpp"


// Libraries for Eigen
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

// Libraries for OpenCV and images
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

// Library for opencv SIFT
// #include <opencv2/features2d.hpp>
// #include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>

// time library
#include <chrono>

// #include "common_functions_library/eigen_typedefs.h"
// #include "common_functions_library/library_header.h"

#include "junction_point_detector/junction_detector.h"
// #include "fishnet_detector_cpp/keypoint_matcher.h"



class FishnetDetectorNode : public rclcpp::Node {
    public:
    FishnetDetectorNode();
    ~FishnetDetectorNode(){};

    // Callback functions for the subscribers
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    // void dist_side_callback(const std_msgs::msg::Float64::SharedPtr msg);
    // void net_ori_side_callback(const geometry_msgs::msg::Quaternion::SharedPtr msg);


    // cv::Mat test_img = cv::imread("/home/docker/net_inspector/auv_ws/test_pool_4_ign.png", cv::IMREAD_COLOR);
    // cv::Mat test2_img = cv::imread("/home/docker/net_inspector/auv_ws/test_pool_5_ign.png", cv::IMREAD_COLOR);
    // void test_loop();
    // bool first_image = true;
    // cv::Mat descriptor_last;
    // PointCloudKeyPoints junction_kpts_last;
    // cv::Mat img_last;
    // double dist_side = 0.0;
    // Eigen::Quaterniond net_ori_side = Eigen::Quaterniond::Identity();


    private:
    // rclcpp::TimerBase::SharedPtr timer_;

    // bool saved_first_image = false;
    // bool saved_second_image = false;
    // bool saved_third_image = false;

    bool publish_image_with_junctions;

    // publisher
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr junction_pub_;

    // subscriber
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    // rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr dist_side_sub_;
    // rclcpp::Subscription<geometry_msgs::msg::Quaternion>::SharedPtr net_ori_side_sub_;
};




#endif // FISHNET_DETECTOR_ROS_H