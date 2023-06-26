#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <opencv2/opencv.hpp>
#include <configs/camera_type.h>

namespace config_utils
{

    bool getBool(std::string s);

    int getInt(std::string s);

    double getDouble(std::string s);

    std::string getString(std::string s);

    cv::Size getCvSize(std::string s);

    CameraType getCameraType(std::string s);

    bool parseConfigFile(std::string path, std::map<std::string, std::string> &conf_map);

}