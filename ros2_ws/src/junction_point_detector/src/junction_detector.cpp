#include "junction_point_detector/junction_detector.h"

void dampenIntensity(cv::Mat& img, double threshold_min, double threshold_max) {
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


PointCloud find_junctions_not_rotated(const cv::Mat &img, const int grid_area, const float grid_area_threshold, const bool show_contours, const int eps, const bool save_images) {

    cv::Mat img_gray;
    cv::Mat img_blur;
    cv::Mat img_thresh;
    cv::Mat img_contours;
    cv::Mat img_contours2;
    cv::Mat img_boxes;
    cv::Mat img_box_with_junctions;
    cv::Mat img_junctions;

    cv::Mat gray;
    // Convert the image to grayscale (1 channel)
    if(img.channels() == 1){
        gray = img.clone();
        if(show_contours || save_images){
            cv::cvtColor(img,img, cv::COLOR_GRAY2BGR);
        }        
    }
    else{
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0);

    cv::Mat thresh;
    cv::adaptiveThreshold(blur, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2); // TODO: Uncomment
    // cv::adaptiveThreshold(blur, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 13, 5); // Used in pool
    if(show_contours || save_images){
        img_gray = gray.clone();
        img_blur = blur.clone();
        img_thresh = thresh.clone();
        img_contours = img.clone();
        img_contours2 = img.clone();
        img_boxes = img.clone();
        img_box_with_junctions = img.clone();
        img_junctions = img.clone();
    }

    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(thresh, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point2f> junctions;

    for (auto &contour : contours) {
        double area = cv::contourArea(contour);

        double estimated_area = grid_area; // Need implementation for grid_area estimation
        if (estimated_area * (1 / (2 * grid_area_threshold)) < area && area < estimated_area * (2 * grid_area_threshold)) {
            if(show_contours || save_images){
                cv::drawContours(img_contours, contours, -1, cv::Scalar(255, 0, 0), 1);
                cv::drawContours(img_contours2, contours, -1, cv::Scalar(255, 0, 0), 1);
            }
            cv::Rect rect = cv::boundingRect(contour);
            double width = rect.width;
            double height = rect.height;
            double box_area = width * height;

            // Calculate the vertices of the rectangle 
            // Shifting points a few pixels out, since assuming the boxes are smaller than the net masks
            int shift = 1;
            cv::Point2f vertices[4];
            vertices[0] = rect.tl() + cv::Point2i(-shift, -shift);
            vertices[1] = cv::Point2f(rect.x + rect.width, rect.y) + cv::Point2f(shift, -shift);
            vertices[2] = rect.br() + cv::Point2i(shift, shift);
            vertices[3] = cv::Point2f(rect.x, rect.y + rect.height) + cv::Point2f(-shift, shift);


            if (area / box_area >= 0.4 && width / height >= 0.5 && width / height <= 2.0) { // For simulator
                // if (area / box_area >= 0.4 && width / height >= 0.65 && width / height <= 1.54) {  // For real
                if (show_contours || save_images){
                    cv::rectangle(img_boxes, rect, cv::Scalar(0, 255, 0), 1);
                    cv::rectangle(img_box_with_junctions, rect, cv::Scalar(0, 255, 0), 1);
                }
                
                for (auto &vertex : vertices) {
                    junctions.push_back(vertex);
                    if (show_contours || save_images){
                        cv::circle(img_box_with_junctions, vertex, 2, cv::Scalar(0, 255, 255), -1);
                    }
                }
            }

            else {
                if(show_contours || save_images){
                    cv::rectangle(img_boxes, rect, cv::Scalar(0, 0, 255), 1);
                }
            }
        }
    }


    if(junctions.size() < 4){
        // return PointCloudKeyPoints();
        return PointCloud();
    }


    KDTree index(2 /* dim */, junctions, 7 /* max leaf */);
    index.index->buildIndex();
    const float radius = eps;
    auto visited = std::vector<bool>(junctions.size());


    std::vector<cv::Point2f> cluster_center;
    // std::vector<cv::KeyPoint> cluster_center_kpts;
    SearchParameters params(10.0,false);

    // Perform radius search for each point
    for (size_t i = 0; i < junctions.size(); ++i) {

        if (visited[i]) continue;

        // Vector to store the indices of neighboring points
        std::vector<nanoflann::ResultItem<size_t, float>> neighbors; // Change the type of neighbors
        neighbors.reserve(6); // Reserve space for efficiency


        // Perform radius search
        index.index->radiusSearch(&junctions[i].x, radius * radius, neighbors, params); //, nanoflann::SearchParams(32, 0.f, false));

        // Check if there are at least 3 neighbors
        if (neighbors.size() >= 3){
            // Add the current point and its neighbors to a new cluster
            std::vector<size_t> cluster;
            cluster.reserve(neighbors.size()); // Reserve space for efficiency
            for (const auto& neighbor : neighbors) {
                cluster.push_back(neighbor.first); // Store the index of the neighboring point
            }

            // Find cluster center
            float x = 0;
            float y = 0;
            for (size_t j = 0; j < cluster.size(); ++j) {
                x += junctions[cluster[j]].x;
                y += junctions[cluster[j]].y;
            }
            x /= cluster.size();
            y /= cluster.size();
            cluster_center.push_back(cv::Point2f(x, y));
            // cluster_center_kpts.emplace_back(cv::Point2f(x, y), 1.0f);
            // cluster_center.push_back(x);

            // Mark all points in the cluster as visited
            for (size_t j = 0; j < cluster.size(); ++j) {
                visited[cluster[j]] = true;
            }

            if(show_contours || save_images){
                cv::circle(img_junctions, cv::Point2f(x, y), 2, cv::Scalar(0, 255, 255), -1);
            }
        } 
            
    }

    if(save_images){
        std::cout << "Saving images..." << std::endl;
        // cv::imwrite("fishnet_detector_imgs/original.jpg", img_original);
        // cv::imwrite("fishnet_detector_imgs/adjusted.jpg", img_adjusted);
        cv::imwrite("../processed_images/gray.png", img_gray);
        cv::imwrite("../processed_images/blur.png", img_blur);
        cv::imwrite("../processed_images/thresh.png", img_thresh);
        cv::imwrite("../processed_images/contours.png", img_contours);
        cv::imwrite("../processed_images/contours2.png", img_contours2);
        cv::imwrite("../processed_images/boxes.png", img_boxes);
        cv::imwrite("../processed_images/boxes_with_junctions.png", img_box_with_junctions);
        cv::imwrite("../processed_images/junctions.png", img_junctions);
    }

    if (show_contours) {
        cv::imshow("Contours", img_contours);
        cv::imshow("Contours2", img_contours2);
        cv::imshow("Boxes", img_boxes);
        cv::imshow("Boxes with junctions", img_box_with_junctions);
        cv::imshow("Junctions", img_junctions);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    return cluster_center; // Return the cluster center
    // return cluster_center_kpts; // Return the cluster center keypoints
}