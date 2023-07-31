#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include "camera_type.h"

namespace config
{

class CalibrationModelParams
{
  public:
    CalibrationModelParams(std::string path) {
        loadParams(path);
    }

    CameraType cam_type;

    cv::Size_<int> img_size;
    cv::Size_<int> board_size;
    double square_size;
    double fov;

    void loadParams(std::string path);
};

}