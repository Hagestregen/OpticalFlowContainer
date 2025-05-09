cmake_minimum_required(VERSION 3.10)
project(Test1Project)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen
find_package(Eigen3 REQUIRED)
# Find OpenCV
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# Include directories
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

# Create executable 
add_executable(junction_detector main.cpp junction_detector.cpp)
# Link OpenCV
target_link_libraries(junction_detector ${OpenCV_LIBS} Threads::Threads)