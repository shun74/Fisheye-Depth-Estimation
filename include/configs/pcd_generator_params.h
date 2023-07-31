#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

namespace config
{

class PointCloudGeneratorParams
{
  public:
    PointCloudGeneratorParams(std::string path) {
        loadParams(path);
    }

    bool is_fisheye;
    
    double x_min, x_max;
    double y_min, y_max;
    double z_min, z_max;

    double fx, fy, cx, cy, base;

    bool use_down_sample;
    double leaf_x, leaf_y, leaf_z;

    void loadParams(std::string path);
};


}