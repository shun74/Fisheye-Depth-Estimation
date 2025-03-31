#pragma once

#include <iostream>
#include <string>

#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/opencv.hpp>

namespace cuda
{

class StereoMatcher
{
  private:
    cv::Ptr<cv::cuda::Filter> blur_filter_;

    cv::Ptr<cv::cuda::StereoSGM> stereo_matcher_;

    bool use_filter_;
    cv::Ptr<cv::cuda::DisparityBilateralFilter> disp_filter_;

  public:
    StereoMatcher(cv::Size blur_kernel, int min_disp, int max_disp, int p1, int p2, int unique_ratio, int mode);

    void setPostFilter(int filter_size, int refine_iter, double edge_thresh, double disc_thresh, double sigma_range);

    void computeDisparity(cv::cuda::GpuMat &d_left, cv::cuda::GpuMat &d_right, cv::cuda::GpuMat &d_disp);
};

} // namespace cuda
