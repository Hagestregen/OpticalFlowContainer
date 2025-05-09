import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


# show image using matplotlib
image_path = "/home/docker/net_inspector/auv_ws/src/auv_common/image_pipeline/junction_point_detector/scripts/junction_point_tests/images/image.png"
image_path = "/home/docker/net_inspector/auv_ws/src/auv_common/image_pipeline/junction_point_detector/scripts/junction_point_tests/images/img_fish1.png"
# image = plt.imread(image_path)
# plt.imshow(image)
# plt.show()


# show image using opencv
image = cv2.imread(image_path)
cv2.imshow("Image", image)
# # Change the blue color to 0
# image[:, :, 0] = 0
# # change the green color to 0
# image[:, :, 1] = 0

#Iterate through each pixel in the image
# for (int y = 0; y < img.rows; ++y) {
#     for (int x = 0; x < img.cols; ++x) {
#         cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);

#         // Calculate the Euclidean distance between the pixel color and the reference color
#         double distance = cv::norm(pixel, referenceColor);

#         // Brighten parts if its part of rope
#         if (distance <= rope_thresh){
#             img.at<cv::Vec3b>(y, x) = cv::Vec3b(
#                 pixel[0] * 1.1,
#                 pixel[1] * 1.1,
#                 pixel[2] * 1.1
#             );
#         }
#         // Dampen the intensity of the pixel based on the distance
#         else {
#             double factor = 1.0 - (distance / 255);
#             factor *= factor; // Square the factor to make the dampening more pronounced
#             img.at<cv::Vec3b>(y, x) = cv::Vec3b(
#                 pixel[0] * factor,
#                 pixel[1] * factor,
#                 pixel[2] * factor
#             );
#         }
#     }
# }

# Create a new array which is just the red channel minus the blue channel
print("Debug1")
print(image[:4, :4, :])

print("Debug2")
print(image[:, :, 0].shape)
print(image.shape)
image_diff = image[:, :, 2] - image[:, :, 0]

# # If image diff is less than -5, then scale all the pixels by the abs value of diff divided by 255
# image_diff[image_diff < -5] = 0
# # If image diff is greater than 5, then scale all the pixels by the abs value of diff divided by 255
# image_diff[image_diff > 5] = 0
# # if image diff is between -5 and 5 scale value
# image_diff[(image_diff >= -5) & (image_diff <= 5)] = image_diff[(image_diff >= -5) & (image_diff <= 5)] / 255

# print(image_diff[:4, :4])

image_r = image[:, :, 2].astype(np.int16)
image_g = image[:, :, 1].astype(np.int16)
image_b = image[:, :, 0].astype(np.int16)

print("Debug3")
print(image_r[:4, :4])
print("Debug4")
print(image_b[:4,:4])

image_diff = image_r - image_b
print("Debug5")
print(image_diff[:4, :4])

image[image_diff < 10] = [0, 0, 0]

# # Iterate through each pixel in the image
# for y in range(image.shape[0]):
#     for x in range(image.shape[1]):
#         pixel = image[y,x]
        
#         # Calculate the Euclidean distance between the pixel color and the reference color
#         reference_color = np.array([0,0,255])
#         distance = np.linalg.norm(pixel - reference_color)
        
#         # Brighten parts if its part of the rope
#         if distance <= 100:
#             image[y,x] = pixel * 1.1
        
#         # Dampen the intensisty of the pixel based on the distance
#         else:
#             factor = 1.0 - (distance / 255.0)
#             factor *= factor
#             # Square the factor 
#             image[y,x] = pixel * factor
            

# # increase the red scale by 5
# image[:, :, 2] = image[:, :, 2] * 5
cv2.imshow("Image changed", image)


# remake the image as a intensity image (grayscale)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# show the image
cv2.imshow("Image gray", gray_image)

cv2.waitKey(0)
cv2.destroyAllWindows()