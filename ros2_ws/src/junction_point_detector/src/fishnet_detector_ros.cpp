#include "junction_point_detector/fishnet_detector_ros.h"

// #include <opencv2/nonfree/features2d.hpp>
// #include <opencv2/features2d.hpp>

// Define the point cloud type
typedef std::vector<cv::Point2f> PointCloud;

FishnetDetectorNode::FishnetDetectorNode() : Node("fishnet_detector_node") {
    // RCLCPP_INFO(this->get_logger(), "Fishnet Detector Node");
    publish_image_with_junctions = false;

    // publishers
    junction_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud>("/junction_detector/junctions", 10);


    if(publish_image_with_junctions){
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/junction_detector/img_with_junctions", 10);
    }

    // subscribers
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        // "/camera/side/infra1/image_rect_raw", 10, std::bind(&FishnetDetectorNode::image_callback, this, std::placeholders::_1));
        // "/camera/side/color/image_raw", 10, std::bind(&FishnetDetectorNode::image_callback, this, std::placeholders::_1));
        "/camera/camera/color/image_raw", 10, std::bind(&FishnetDetectorNode::image_callback, this, std::placeholders::_1));
}


void FishnetDetectorNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    // RCLCPP_INFO(this->get_logger(), "Image received");

    // auto start = std::chrono::high_resolution_clock::now();

    auto bridge = cv_bridge::CvImage::Ptr();
    try {
        // std::cout << "Image encoding: " << msg->encoding << std::endl;
        bridge = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8); // For color images
        // bridge = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8); // For color images
        // bridge = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8); // For grayscale images
        // std::cout << "Image encoding2: " << bridge->encoding << std::endl;
    } catch (cv_bridge::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat img = bridge->image;
    cv::Mat img_with_junctions = img.clone();

    dampenIntensity(img, -20, 15);

    // if(saved_first_image == false) {
    //     cv::imwrite("/home/docker/net_inspector/auv_ws/test_img_1.png", img);
    //     saved_first_image = true;
    // }


    // PointCloudKeyPoints junction_kpts = find_junctions_not_rotated(img, 200, 2.0, false, 6);
    PointCloud junction_pts = find_junctions_not_rotated(img, 200, 2.0, false, 6);
    // if(junction_kpts.size() < 4) {
    if(junction_pts.size() < 4) {
        RCLCPP_INFO(this->get_logger(), "No junctions found");
        return;
    }
    
    // Create a PointCloud message
    sensor_msgs::msg::PointCloud junction_msg;
    junction_msg.header.stamp = msg->header.stamp;
    junction_msg.header.frame_id = msg->header.frame_id;
    junction_msg.points.reserve(junction_pts.size());
    // std::cout << "Junctions found: " << junction_pts.size() << std::endl;
    for(const auto& junction : junction_pts) {
        geometry_msgs::msg::Point32 point;
        point.x = junction.x;
        point.y = junction.y;
        point.z = 0.0; // Assuming z is 0 for 2D points
        junction_msg.points.push_back(point);

        // std::cout << "Junction point: " << junction.x << ", " << junction.y << std::endl;
    }
    junction_pub_->publish(junction_msg);

    if(publish_image_with_junctions){
        // cv::Mat img_with_junctions = img.clone();
        for(const auto& junction : junction_pts) {
            cv::circle(img_with_junctions, junction, 2, cv::Scalar(200, 200, 0), -1);
            // cv::circle(img_with_junctions, junction_kpts[i].pt, 2, cv::Scalar(200, 200, 0), -1);
        }
    
        // auto img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img_with_junctions).toImageMsg();
        auto img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "rgb8", img_with_junctions).toImageMsg();
        image_pub_->publish(*img_msg);
    }
}