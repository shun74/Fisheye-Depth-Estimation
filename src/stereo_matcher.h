#pragma once

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

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
    StereoMatcher(bool gray_scale, std::string algorithm, cv::Size blur_kernel, int max_disp, int block_size);

    StereoMatcher(bool gray_scale, std::string algorithm, cv::Size blur_kernel, int min_disp, int max_disp,
                  int block_size, int p1, int p2, int max_diff, int pre_fc, int unique_ratio, int speckle_size,
                  int speckle_range, int mode);

    void setPostFilter(float lambda, float sigma);

    void computeDisparity(const cv::Mat &left, const cv::Mat &right, cv::Mat &disp);
};
