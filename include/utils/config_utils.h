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


    CameraType getCameraType(std::string s);

    bool parseConfigFile(std::string path, std::map<std::string, std::string> &conf_map);

    std::map<std::string, std::string> parseArguments(int argc, char **argv);

    template<typename T>
    cv::Size_<T> getCvSize(std::string s) {
        // "[100,200]" -> cv::Size(100, 200)
        s.erase(std::remove(s.begin(), s.end(), '['), s.end());
        s.erase(std::remove(s.begin(), s.end(), ']'), s.end());

        std::stringstream ss(s);
        std::vector<std::string> num_s;
        while (std::getline(ss, s, ','))
        {
            num_s.push_back(s);
        }
        
        if (std::is_same<T, int>::value) {
            return cv::Size_<T>(std::stoi(num_s[0]), std::stoi(num_s[1]));
        }
        else if (std::is_same<T, double>::value) {
            return cv::Size_<T>(std::stod(num_s[0]), std::stod(num_s[1]));
        }
    }
}