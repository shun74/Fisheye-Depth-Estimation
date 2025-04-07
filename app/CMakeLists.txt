cmake_minimum_required(VERSION 3.18)
project(FisheyeDepthEstimation_apps LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(VTK REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
endif()

add_executable(calibration calibration.cpp)
target_link_libraries(calibration FisheyeDepthEstimation_lib ${OpenCV_LIBS} OpenMP::OpenMP_CXX yaml-cpp)

add_executable(compare_fisheye compare_fisheye.cpp)
target_link_libraries(compare_fisheye FisheyeDepthEstimation_lib ${OpenCV_LIBS} OpenMP::OpenMP_CXX yaml-cpp)

add_executable(single_test single_test.cpp)
target_link_libraries(single_test FisheyeDepthEstimation_lib ${OpenCV_LIBS} OpenMP::OpenMP_CXX yaml-cpp)

add_executable(realtime_stereo realtime_stereo.cpp)
target_link_libraries(realtime_stereo FisheyeDepthEstimation_lib ${OpenCV_LIBS} ${VTK_LIBRARIES} OpenMP::OpenMP_CXX yaml-cpp)

add_executable(realtime_stereo_cuda realtime_stereo_cuda.cpp)
target_link_libraries(realtime_stereo_cuda FisheyeDepthEstimation_cuda_lib ${OpenCV_LIBS} ${VTK_LIBRARIES} OpenMP::OpenMP_CXX yaml-cpp)
