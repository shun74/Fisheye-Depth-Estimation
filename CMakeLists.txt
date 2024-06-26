cmake_minimum_required(VERSION 3.0)
project(fisheye_stereo)

option(ENABLE_CUDA "enable cuda build" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

include_directories(include libs ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

set(COMMON_SOURCES
    src/utils/config_utils.cpp
    src/utils/undistort_utils.cpp
)

add_library(stereo_matcher
    src/engines/stereo_matcher.cpp
    src/configs/stereo_matcher_params.cpp
    ${COMMON_SOURCES}
)
target_link_libraries(stereo_matcher ${OpenCV_LIBS} stdc++fs)

add_library(pcd_generator
    src/engines/pcd_generator.cpp
    src/configs/pcd_generator_params.cpp
    ${COMMON_SOURCES}
)
target_link_libraries(pcd_generator ${OpenCV_LIBS} ${PCL_LIBRARIES} stdc++fs)

add_executable(calibration
    app/calibration.cpp
    src/engines/calibration_model.cpp
    src/configs/calibration_model_params.cpp
    ${COMMON_SOURCES}
)
target_link_libraries(calibration ${OpenCV_LIBS})

add_executable(realtime_stereo
    app/realtime_stereo.cpp
    src/stereo_vision.cpp
    src/configs/stereo_vision_params.cpp
    ${COMMON_SOURCES}
)
target_link_libraries(realtime_stereo stereo_matcher pcd_generator ${OpenCV_LIBS} ${PCL_LIBRARIES} stdc++fs)

add_executable(test
    app/test.cpp
    ${COMMON_SOURCES}
)
target_link_libraries(test stereo_matcher pcd_generator ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(compare_fisheye
    app/compare_fisheye.cpp
    ${COMMON_SOURCES}
)
target_link_libraries(compare_fisheye stereo_matcher pcd_generator ${OpenCV_LIBS} ${PCL_LIBRARIES})

if(ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIBRARY_DIRS})

    set(CMAKE_CUDA_ARCHITECTURES "86")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode arch=compute_86,code=sm_86")

    add_executable(realtime_stereo_cuda
        app/realtime_stereo_cuda.cpp
        src/stereo_vision_cuda.cpp
        src/configs/stereo_vision_params.cpp
        src/engines/cuda/stereo_matcher_cuda.cpp
        src/engines/cuda/pcd_generator_cuda.cpp
        src/configs/stereo_matcher_params.cpp
        src/engines/pcd_generator.cpp
        src/configs/pcd_generator_params.cpp
        ${COMMON_SOURCES}
    )

    target_link_libraries(realtime_stereo_cuda ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CUDA_LIBRARIES})
endif()
