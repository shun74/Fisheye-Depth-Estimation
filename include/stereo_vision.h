#pragma once

#include <iostream>
#include <shared_mutex>
#include <thread>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/io.h>
#include <configs/stereo_vision_params.h>
#include <engines/pcd_generator.h>
#include <engines/stereo_matcher.h>


class StereoVisionProcessor
{
  private:
    // data to share among threads 
    std::shared_mutex mtx_img_, mtx_disp_, mtx_pcd_;
    cv::Mat img_, left_, right_, disp_, rgb_;
    pcl::PointCloud<pcl::PointXYZRGB> pcd_;

    // thread & stop flag
    std::thread th_img_update_, th_disp_, th_pcd_, th_viewer_, th_pcd_viewer_;
    bool stop_img_update_, stop_disp_, stop_pcd_, stop_viewer_, stop_pcd_viewer_;
    bool up_img_update_, up_disp_, up_pcd_, up_viewer_, up_pcd_viewer_;
    bool video_cap_;
    
    // basic params
    std::vector<cv::Mat> left_maps_, right_maps_;
    cv::Size img_size_;
    int img_update_sleep_;

    // engine
    engine::StereoMatcher stereo_matcher_;
    engine::PointCloudGenerator pcd_generator_;

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
    StereoVisionProcessor(std::string path, std::vector<cv::Mat> left_maps, std::vector<cv::Mat> right_maps);

    ~StereoVisionProcessor() {
        stopThreads();
    }

    void setParams(config::StereoVisionParams sv_params);

    void setConvertMaps(cv::Mat map_x, cv::Mat map_y);

    bool run(int video_cap=0);
};