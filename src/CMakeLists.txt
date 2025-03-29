cmake_minimum_required(VERSION 3.18)
set(PROJECT_NAME FisheyeDepthEstimation_lib)
set(INTERFACE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include)

project(${PROJECT_NAME} LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

file(GLOB SRCS *.cpp *.h*)
add_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE ${SRCS})
target_include_directories(${PROJECT_NAME} 
    PRIVATE 
        ${INTERNAL_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    PUBLIC
        ${INTERFACE_INCLUDE_DIR}
)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} 
    PUBLIC 
        ${OpenCV_LIBS} 
        OpenMP::OpenMP_CXX
)
set_target_properties(${PROJECT_NAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${INTERFACE_INCLUDE_DIR})

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wall -O3>
    # $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wall -g>
    $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wall -O3>
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4819>
)
