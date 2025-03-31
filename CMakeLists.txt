cmake_minimum_required(VERSION 3.10)
project(FisheyeDepthEstimation)

option(ENABLE_CUDA "enable cuda build" OFF)
set(CUDA_ARCHS "75;86")

add_subdirectory(src)
add_subdirectory(app)
