#include "junction_detector.h"



PointCloudKeyPoints find_contours(const cv::Mat &img, const int grid_area, const float grid_area_threshold, const bool show_contours, const int eps, const bool save_images) {

    // cv::Mat img_original = img.clone();
    // cv::Mat img_adjusted = img.clone();
    cv::Mat img_gray;
    cv::Mat img_blur;
    cv::Mat img_thresh;
    cv::Mat img_contours;
    cv::Mat img_contours2;
    cv::Mat img_boxes;
    cv::Mat img_box_with_junctions;
    cv::Mat img_junctions;

    // Average value of a slice of the image [0:4, 0:4]
    // cv::Scalar avg = cv::mean(img(cv::Rect(0, 0, 4, 4)));
    // std::cout << "Average value: " << avg << std::endl;

    // avg = cv::mean(img(cv::Rect(6, 5, 4, 4)));
    // std::cout << "Average value: " << avg << std::endl;

    // // First pixel of image:
    // std::cout << "First pixel value: " << img.at<cv::Vec3b>(0, 0) << std::endl;

    // std::cout << "Image shape (rows, cols, channels): " << img.rows << " " << img.cols << " " << img.channels() << std::endl;
    
    // // Adjust the colors of the image
    // adjustColors5(img_adjusted);
    // dampenIntensity(img_adjusted, cv::Vec3b(83, 79, 118), 30.0);



    cv::Mat gray;
    if(img.channels() == 1){
        gray = img.clone();
        if(show_contours || save_images){
            cv::cvtColor(img,img, cv::COLOR_GRAY2BGR);
        }        
    }
    else{
        // Convert the image to grayscale (1 channel)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    }

    // if (show_contours  || save_images) {
    //     img_gray = img.clone();
    //     img_blur = img.clone();
    //     img_thresh = img.clone();
    //     img_contours = img.clone();
    //     img_contours2 = img.clone();
    //     img_boxes = img.clone();
    //     img_box_with_junctions = img.clone();
    //     img_junctions = img.clone();
    // }

    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0);
    // cv::GaussianBlur(gray, blur, cv::Size(1, 1), 0);
    // cv::GaussianBlur(blur, blur, cv::Size(3, 3), 0);

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
    // Take the time of 100 iterations
    // auto start = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < 100; ++i) {
    //     cv::findContours(thresh, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    // }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::cout << "Time taken for 100 iterations: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // start = std::chrono::high_resolution_clock::now();
    // for(int i = 0; i < 100; ++i){
    //     std::vector<cv::Vec4i> hierarchy; // TODO: remove hierarchy if not needed
    //     cv::findContours(thresh, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    // }
    // end = std::chrono::high_resolution_clock::now();
    // std::cout << "Time taken for 100 iterations with hierarchy: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    cv::findContours(thresh, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    // std::vector<cv::Vec4i> hierarchy; // TODO: remove hierarchy if not needed
    // cv::findContours(thresh, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point2f> junctions;

    // float avg_box_area = 0;
    // int i = 0;
    double avrg_angle = 0;
    int num_angles = 0;
    for (auto &contour : contours) {
        double area = cv::contourArea(contour);
        // cv::Moments M = cv::moments(contour);
        // cv::Point2f contour_center;
        
        // if (M.m00 != 0) {
        //     contour_center = cv::Point2f(static_cast<float>(M.m01 / M.m00), static_cast<float>(M.m10 / M.m00));
        // } else {
        //     contour_center = cv::Point2f(0, 0);
        // }


        double estimated_area = grid_area; // Need implementation for grid_area estimation
        if (estimated_area * (1 / (2 * grid_area_threshold)) < area && area < estimated_area * (2 * grid_area_threshold)) {
            if(show_contours || save_images){
                cv::drawContours(img_contours, contours, -1, cv::Scalar(255, 0, 0), 1);
                cv::drawContours(img_contours2, contours, -1, cv::Scalar(255, 0, 0), 1);
            }

            cv::RotatedRect rect = cv::minAreaRect(contour);

            // rect.angle = 20;
            cv::Point2f vertices[4];
            rect.points(vertices);


            double width = rect.size.width;
            double height = rect.size.height;
            double box_area = width * height;

            // i++;
            // avg_box_area += box_area;

            if (rect.angle < -45) {
                rect.angle += 90;
                std::swap(width, height);
            }
            
            
            // TODO: add constrain on the angle
            if (area / box_area >= 0.4 && width / height >= 0.5 && width / height <= 2.0) { // For simulator
            // if (area / box_area >= 0.4 && width / height >= 0.65 && width / height <= 1.54) {  // For real
                if (show_contours || save_images){
                    // cv::rectangle(img_boxes, rect.boundingRect(), cv::Scalar(0, 255, 0), 1);
                    // cv::rectangle(img_box_with_junctions, rect.boundingRect(), cv::Scalar(0, 255, 0), 1);

                    // Draw roted boxes using lines:
                    for (int j = 0; j < 4; ++j) {
                        cv::line(img_boxes, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 1);
                        cv::line(img_box_with_junctions, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 1);
                    }
                }
                // cv::Point2f center = rect.center;
                for (auto &vertex : vertices) {

                    // cv::Point2f vec = vertex - center;
                    // cv::Point2f vertex_scaled = center + vec * 1.06;
                    junctions.push_back(vertex);
                    if (show_contours || save_images){
                        cv::circle(img_box_with_junctions, vertex, 2, cv::Scalar(0, 255, 255), -1);
                    }
                }

                avrg_angle += rect.angle;
                num_angles++;

            }
            else{
                if(show_contours || save_images){
                    // cv::rectangle(img_boxes, rect.boundingRect(), cv::Scalar(0, 0, 255), 1);

                    // Draw roted boxes using lines:
                    for (int j = 0; j < 4; ++j) {
                        cv::line(img_boxes, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 0, 255), 1);
                    }
                }
            }
        }
    }

    if(num_angles > 0){
        avrg_angle /= num_angles;
        std::cout << "Average angle: " << avrg_angle << std::endl;
    }

    // avg_box_area /= i;
    // std::cout << "Average box area: " << avg_box_area << std::endl;


    // std::vector<std::vector<size_t>> clusters;
    // std::vector<cv::Point2f> cluster_center;



    if(junctions.size() < 4){
        return PointCloudKeyPoints();
    }


    KDTree index(2 /* dim */, junctions, 7 /* max leaf */);
    index.index->buildIndex();
    const float radius = eps;
    auto visited = std::vector<bool>(junctions.size());


    std::vector<cv::Point2f> cluster_center;
    std::vector<cv::KeyPoint> cluster_center_kpts;
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
            // std::vector<size_t> cluster(neighbors.begin(), neighbors.end());
            // cluster.push_back(i); // Add the current point

            // Add the current point and its neighbors to a new cluster
            std::vector<size_t> cluster;
            cluster.reserve(neighbors.size()); // Reserve space for efficiency
            for (const auto& neighbor : neighbors) {
                // std::cout << "Neighbor index: " << neighbor.first << std::endl;
                // std::cout << "Neighbor distance: " << neighbor.second << std::endl;
                cluster.push_back(neighbor.first); // Store the index of the neighboring point
            }

            // clusters.push_back(cluster); // TODO: change it to only store the center of the cluster

            // Find cluster center
            float x = 0;
            float y = 0;
            for (size_t j = 0; j < cluster.size(); ++j) {
                x += junctions[cluster[j]].x;
                y += junctions[cluster[j]].y;
            }
            x /= cluster.size();
            y /= cluster.size();
            // cluster_center.push_back(cv::Point2f(x, y));
            cluster_center_kpts.emplace_back(cv::Point2f(x, y), 1.0f);
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

    if (save_images) {
        cv::imwrite("fishnet_detector_imgs/junctions.jpg", img_junctions);
    }

    return cluster_center_kpts; // Return the cluster center
}




PointCloudKeyPoints find_contours_not_rotated(const cv::Mat &img, const int grid_area, const float grid_area_threshold, const bool show_contours, const int eps, const bool save_images) {

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
        return PointCloudKeyPoints();
    }


    KDTree index(2 /* dim */, junctions, 7 /* max leaf */);
    index.index->buildIndex();
    const float radius = eps;
    auto visited = std::vector<bool>(junctions.size());


    std::vector<cv::Point2f> cluster_center;
    std::vector<cv::KeyPoint> cluster_center_kpts;
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
            // cluster_center.push_back(cv::Point2f(x, y));
            cluster_center_kpts.emplace_back(cv::Point2f(x, y), 1.0f);
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

    return cluster_center_kpts; // Return the cluster center
}