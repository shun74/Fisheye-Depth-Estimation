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
        std::string algorithm_;

        cv::Size blur_kernel_;
        
        cv::Ptr<cv::cuda::StereoSGM> stereo_matcher_;

        bool use_filter_;
        cv::Ptr<cv::cuda::DisparityBilateralFilter> disp_filter_; 


    public:
        StereoMatcher(std::string path);
        StereoMatcher(config::StereoMatcherParams sm_params);

        void setParameters(config::StereoMatcherParams sm_params);

        void computeDisparity(cv::Mat &left, cv::Mat &right, cv::Mat &disp);
};

}

}