// #include <iostream>
// #include <eigen3/Eigen/Dense>
// #include <eigen3/Eigen/Core>
// #include <opencv2/opencv.hpp>
#include "junction_detector.h"
#include "adjust_color.h"
using namespace std;

int main() {


    // Load an image with openCV
    // cv::Mat img = cv::imread("images/image_622.png", cv::IMREAD_GRAYSCALE);
    // cv::Mat img = cv::imread("/home/docker/net_inspector/auv_ws/src/auv_common/image_pipeline/junction_point_detector/scripts/junction_point_tests/images/image_622.png");
    
    // Rel path from build folder
    // cv::Mat img = cv::imread("../images/image.png");
    cv::Mat img = cv::imread("../images/img_fish2.png");
    // cv::Mat img = cv::imread("../images/image_622.png");
    if (img.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }
    
    // Print image shape
    std::cout << "Image shape (row, cols, channels):" << img.rows << " " << img.cols << " " << img.channels() << std::endl;

    // // resize image to 848x480x3
    // cv::resize(img, img, cv::Size(848, 480));
    // std::cout << "Resized shape (row, cols, channels):" << img.rows << " " << img.cols << " " << img.channels() << std::endl;

    cv::imshow("Original Image", img);

    // // Print the BGR values of the right bottom corner pixels (10x10)
    // for (int i = img.rows - 30; i < img.rows; ++i) {
    //     for (int j = img.cols - 30; j < img.cols; ++j) {
    //         cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
    //         std::cout << "Pixel at (" << i << ", " << j << "): B=" << (int)pixel[0] << ", G=" << (int)pixel[1] << ", R=" << (int)pixel[2] << std::endl;
    //     }
    // }

    // adjustColors(img, 1.1);
    // adjustColors5(img);
    // dampenIntensity(img, cv::Vec3b(100, 137, 180), 30.0);
    dampenIntensity2(img, -20, 15);
    // cv::imshow("Adjusted Image", img);
    // cv::Mat img_gray;
    // cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    // cv::imshow("Gray Image", img_gray);
    // cv::Mat img_blur;
    // cv::GaussianBlur(img_gray, img_blur, cv::Size(3, 3), 0);
    // cv::imshow("Blurred Image", img_blur);
    // cv::Mat img_thresh;
    // cv::adaptiveThreshold(img_blur, img_thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
    // cv::imshow("Thresholded Image", img_thresh);
    // cv::waitKey(0);
    // cv::destroyAllWindows();


    find_contours_not_rotated(img, 255, 2, false, 6, false);

    return 0;
}