cmake_minimum_required(VERSION 3.18)
set(PROJECT_NAME FisheyeDepthEstimation_lib)
set(INTERFACE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include)
set(CUDA_PROJECT_NAME FisheyeDepthEstimation_cuda_lib)

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
    $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wall -O3>
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4819>
)

if(ENABLE_CUDA)
    enable_language(CUDA)
    
    find_package(CUDAToolkit REQUIRED)
    find_package(OpenCV REQUIRED)
    
    message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
    
    file(GLOB CUDA_SRCS cuda/*.cpp cuda/*.h cuda/*.cu)
    
    add_library(${CUDA_PROJECT_NAME})
    target_sources(${CUDA_PROJECT_NAME} PRIVATE ${CUDA_SRCS} undistort.cpp)
    
    target_include_directories(${CUDA_PROJECT_NAME} 
        PRIVATE 
            ${INTERNAL_INCLUDE_DIR}
            ${OpenCV_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}/cuda
        PUBLIC
            ${INTERFACE_INCLUDE_DIR}
    )
    
    target_compile_features(${CUDA_PROJECT_NAME} PRIVATE cxx_std_17)
    set_target_properties(${CUDA_PROJECT_NAME} PROPERTIES 
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${CUDA_ARCHS}"
    )
    
    target_link_libraries(${CUDA_PROJECT_NAME} 
        PUBLIC 
            ${OpenCV_LIBS} 
            CUDA::cudart
    )
    
    target_compile_options(${CUDA_PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-Wall -O3>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:Clang>:-Wall -O3>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/wd4819>>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
    )
    
    target_compile_definitions(${CUDA_PROJECT_NAME} PUBLIC -DHAVE_CUDA)
endif()