#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

namespace config
{

class StereoVisionParams
{
  public:
    StereoVisionParams(std::string path) {
        loadParams(path);
    }

    cv::Size_<int> img_size;
    cv::Size_<int> win_size;

    int img_update_sleep;
    int viewer_update_sleep;

    bool source_viewer;
    bool rectified_viewer;
    bool disparity_viewer;
    bool point_cloud_viewer;

    int points_size;
    double coordinate_system;

    void loadParams(std::string path);
};

} // namespace config
