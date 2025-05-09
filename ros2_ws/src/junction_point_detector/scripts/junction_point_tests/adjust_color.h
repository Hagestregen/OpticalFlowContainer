#pragma once


#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

inline void adjustColors(cv::Mat& img, double threshold) {
    CV_Assert(img.channels() == 3); // Ensure the image is in BGR format

    // Split the image into individual channels
    std::vector<cv::Mat> channels;
    cv::split(img, channels);

    // Create a mask where blue values are significantly greater than red values
    cv::Mat mask = (channels[0] + 0.3*channels[1] > threshold * channels[2]);

    // Reduce the intensity of all channels where the mask is true
    for (int i = 0; i < img.channels(); ++i) {
        cv::Mat blend = channels[i] * 0.3;
        // channels[i].copyTo(blend, mask);
        blend.copyTo(channels[i], mask);
        // channels[i].setTo(channels[i] * 0.1, mask);
    }

    // Merge the channels back into the image
    cv::merge(channels, img);
}


// void adjustColors2(cv::Mat& img){
//     // Dampen the intensity of a pixel by how far it is away from [83, 79, 118] 

//     for (int i = 0; i < img.rows; ++i) {
//         for (int j = 0; j < img.cols; ++j) {
//             cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
//             cv::Vec3b target_color(83, 79, 118);
//             cv::Vec3b diff = pixel - target_color;
//             cv::Vec3b diff_squared;
//             cv::multiply(diff, diff, diff_squared);
//             cv::Scalar diff_squared_sum = cv::sum(diff_squared);
//             cv::Mat dampening_factor = 1 - diff_squared_sum / 255.0;

//             img.at<cv::Vec3b>(i, j) = pixel.mul(dampening_factor);
//         }
//     }

//     // cv::Vec3b target_color(83, 79, 118);
//     // cv::Mat target_color_mat(1, 1, CV_8UC3, target_color);

//     // cv::Mat img_mat = img.reshape(1, img.total());
//     // cv::Mat diff = img_mat - target_color_mat;
//     // cv::Mat diff_squared;
//     // cv::multiply(diff, diff, diff_squared);
//     // cv::Scalar diff_squared_sum = cv::sum(diff_squared);
//     // cv::Mat dampening_factor = 1 - diff_squared_sum / 255.0;

//     // cv::Mat diff_squared_sum_sqrt;
//     // cv::sqrt(diff_squared_sum, diff_squared_sum_sqrt);
//     // cv::Mat dampening_factor = 1 - diff_squared_sum_sqrt / 255.0;


// }

inline void adjustColors3(cv::Mat& img){
    CV_Assert(img.channels() == 3); // Ensure the image is in BGR format

    // Split the image into individual channels
    std::vector<cv::Mat> channels;
    cv::split(img, channels);

    // Create a mask where blue values are significantly greater than red values
    cv::Mat mask = (channels[0] > channels[2] + 5 | channels[1] > channels[2] + 5 );

    // Reduce the intensity of all channels where the mask is true
    for (int i = 0; i < img.channels(); ++i) {
        cv::Mat blend = channels[i] * 0.3;
        // channels[i].copyTo(blend, mask);
        blend.copyTo(channels[i], mask);
        // channels[i].setTo(channels[i] * 0.1, mask);
    }

    // Merge the channels back into the image
    cv::merge(channels, img);
}

inline void adjustColors4(cv::Mat& img){
    CV_Assert(img.channels() == 3); // Ensure the image is in BGR format

    // Split the image into individual channels
    std::vector<cv::Mat> channels;
    cv::split(img, channels);

    // Create a mask where blue values are significantly greater than red values
    cv::Mat mask = (channels[0] > channels[2] + 7) | (channels[1] > channels[2] + 7) &  ~(channels[0] <= 97 & channels[1] <= 97 &  channels[2] >= 52); // | 
    
    // cv::Mat mask = (channels[0] <= 97 & channels[1] <= 97 &  channels[2] >= 52);

    // Reduce the intensity of all channels where the mask is true
    for (int i = 0; i < img.channels(); ++i) {
        cv::Mat blend = channels[i] * 0.5;
        // channels[i].copyTo(blend, mask);
        blend.copyTo(channels[i], mask);
        // channels[i].setTo(channels[i] * 0.1, mask);
    }

    // Merge the channels back into the image
    cv::merge(channels, img);
}

inline void adjustColors5(cv::Mat& img){
    CV_Assert(img.channels() == 3); // Ensure the image is in BGR format

    // Split the image into individual channels
    std::vector<cv::Mat> channels;
    cv::split(img, channels);

    channels[0] = channels[0] * 0.5;
    channels[1] = channels[1] * 0.7;
    channels[2] = channels[2] * 1.3;

    // Merge the channels back into the image
    cv::merge(channels, img);
}

inline void dampenIntensity(cv::Mat& img, const cv::Vec3b& referenceColor, double rope_thresh) {
    CV_Assert(img.channels() == 3); // Ensure the image is in BGR format

    // Iterate through each pixel in the image
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);

            // Calculate the Euclidean distance between the pixel color and the reference color
            double distance = cv::norm(pixel, referenceColor);

            // Brighten parts if its part of rope
            if (distance <= rope_thresh){
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    pixel[0] * 1.1,
                    pixel[1] * 1.1,
                    pixel[2] * 1.1
                );
            }
            // Dampen the intensity of the pixel based on the distance
            else {
                double factor = 1.0 - (distance / 255);
                factor *= factor; // Square the factor to make the dampening more pronounced
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    pixel[0] * factor,
                    pixel[1] * factor,
                    pixel[2] * factor
                );
            }
        }
    }
}


inline void dampenIntensity2(cv::Mat& img, double threshold_min, double threshold_max) {
    CV_Assert(img.channels() == 3); // Ensure the image is in BGR format

    double incline = 1.0 / (threshold_max - threshold_min);
    double intercept = -threshold_min * incline;

    // Iterate through each pixel in the image
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);

            // diff between red and blue
            double diff = pixel[2] - pixel[0];

            double gain = diff * incline + intercept;
            gain = std::max(std::min(gain, 1.0), 0.0); // Clamp gain to [0, 1]

            // Apply the gain to the pixel
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                pixel[0] * gain,
                pixel[1] * gain,
                pixel[2] * gain
            );
        }
    }
}

