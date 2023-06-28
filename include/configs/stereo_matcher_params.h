#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

namespace config
{

class StereoMatcherParams
{
    public:
        StereoMatcherParams(std::string path) {
            loadParams(path);
        }

        bool gray_scale;

        std::string algorithm;

        cv::Size_<int> blur_kernel;

        int block_size, min_disp, max_disp;
        int p1, p2, max_diff, pre_fc;
        int speckle_size, speckle_range;
        int unique_ratio, mode;

        void loadParams(std::string path);
};

}