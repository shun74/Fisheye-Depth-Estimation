#pragma once

#include <iostream>
#include <shared_mutex>
#include <thread>
#include <atomic>
#include <future>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/io.h>
#include <configs/stereo_vision_params.h>
#include <engines/cuda/pcd_generator_cuda.h>
#include <engines/cuda/stereo_matcher_cuda.h>


namespace cuda
{

class StereoVisionProcessor
{
  private:
    // Image update thread
    std::shared_mutex mtx_img_;
    cv::Mat img_, left_, right_;
    std::atomic<bool> stop_img_update_thread_;
    std::atomic<bool> is_video_cap_open_;
    std::future<void> fut_img_update_thread_;

    // Compute disparity thread
    std::shared_mutex mtx_disp_;
    cv::Mat disp_;
    std::atomic<bool> stop_disp_thread_;
    std::future<void> fut_disp_thread_;

    // Compute point cloud thread
    std::shared_mutex mtx_pcd_;
    cv::Mat rgb_;
    pcl::PointCloud<pcl::PointXYZRGB> pcd_;
    std::atomic<bool> stop_pcd_thread_;
    std::future<void> fut_pcd_thread_;

    // Images Viewer thread
    std::atomic<bool> stop_viewer_thread_;
    std::future<void> fut_viewer_thread_;

    // PCD Viewer thread
    std::atomic<bool> stop_pcd_viewer_thread_;
    std::future<void> fut_pcd_viewer_thread_;

    
    // basic params
    std::vector<cv::Mat> left_maps_, right_maps_;
    cv::Size img_size_;
    int img_update_sleep_;

    // engine
    engine::cuda::StereoMatcher stereo_matcher_;
    engine::cuda::PointCloudGenerator pcd_generator_;

    // viewer params
    int viewer_update_sleep_;
    cv::Size win_size_;
    bool source_viewer_, rectified_viewer_, disparity_viewer_, point_cloud_viewer_;

    // pcd viewer
    int points_size_;
    double coordinate_system_;

    bool is_fisheye_;

    void updateImage(int video_cap);

    void computeDisparity();

    void computePointCloud();

    void updateViewer();

    void updatePcdViewer();

    bool stopThreads();

  public:
    StereoVisionProcessor(std::string config_path, std::vector<cv::Mat> left_maps, std::vector<cv::Mat> right_maps);

    ~StereoVisionProcessor() {
        stopThreads();
    }

    void setParams(config::StereoVisionParams sv_params);

    void setConvertMaps(cv::Mat map_x, cv::Mat map_y);

    bool run(int video_cap=0);
};

} // namespace cuda
