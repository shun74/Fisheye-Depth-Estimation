#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <configs/stereo_matcher_params.h>

namespace engine
{

namespace cuda
{

class StereoMatcher
{
    private:
        cv::Ptr<cv::cuda::StereoSGM> stereo_matcher_;

        std::string algorithm_;

        cv::Size blur_kernel_;

    public:
        StereoMatcher(std::string path);
        StereoMatcher(config::StereoMatcherParams sm_params);

        void setParameters(config::StereoMatcherParams sm_params);

        void computeDisparity(cv::Mat &left, cv::Mat &right, cv::Mat &disp);
};

}

}