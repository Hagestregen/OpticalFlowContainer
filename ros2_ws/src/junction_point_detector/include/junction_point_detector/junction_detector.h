#ifndef JUNCTION_DETECTOR_H
#define JUNCTION_DETECTOR_H


#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#include "junction_point_detector/vendor/nanoflann/nanoflann.hpp"
// #include "fishnet_detector_cpp/KDTreeVectorOfVectorsAdaptor.h"
#include "junction_point_detector/KDTreeVectorOfCVPoint2fAdaptor.h"

using namespace nanoflann;

// Define the point cloud type
typedef std::vector<cv::Point2f> PointCloud;
typedef std::vector<cv::KeyPoint> PointCloudKeyPoints;
// typedef KDTreeVectorOfVectorsAdaptor<PointCloud, float> KDTree;
typedef KDTreeVectorOfVectorsAdaptor<PointCloud, float> KDTree;


void dampenIntensity(cv::Mat& img, double threshold_min, double threshold_max);
// PointCloudKeyPoints find_junctions_not_rotated(const cv::Mat &img, const int grid_area = 250, const float grid_area_threshold = 2, const bool show_contours = false, const int eps = 4, const bool save_images = false);
PointCloud find_junctions_not_rotated(const cv::Mat &img, const int grid_area = 250, const float grid_area_threshold = 2, const bool show_contours = false, const int eps = 4, const bool save_images = false);


#endif // JUNCTION_DETECTOR_H