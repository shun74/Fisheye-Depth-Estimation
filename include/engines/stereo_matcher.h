#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <configs/stereo_matcher_params.h>

namespace engine
{

class StereoMatcher
{
  private:
    cv::Ptr<cv::StereoMatcher> stereo_matcher_;
    
    bool gray_scale_;

    std::string algorithm_;

    cv::Size blur_kernel_;

    // wls filtering
    bool use_filter_;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter_;
    cv::Ptr<cv::StereoMatcher> stereo_matcher_r_;


  public:
    StereoMatcher(std::string path);
    StereoMatcher(config::StereoMatcherParams sm_params);

    void setParameters(config::StereoMatcherParams sm_params);

    void computeDisparity(cv::Mat &left, cv::Mat &right, cv::Mat &disp);
};

} // naemspace engine
