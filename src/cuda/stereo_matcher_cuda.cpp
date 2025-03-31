#include "stereo_matcher_cuda.h"

namespace cuda
{

StereoMatcher::StereoMatcher(cv::Size blur_kernel, int min_disp, int max_disp, int p1, int p2, int unique_ratio,
                             int mode)
    : use_filter_(false)
{
    blur_filter_ = cv::cuda::createBoxFilter(CV_8UC1, CV_8UC1, blur_kernel);
    stereo_matcher_ = cv::cuda::createStereoSGM(min_disp, max_disp, p1, p2, unique_ratio, mode);
}

void StereoMatcher::setPostFilter(int filter_size, int refine_iter, double edge_thresh, double disc_thresh,
                                  double sigma_range)
{
    use_filter_ = true;
    disp_filter_ = cv::cuda::createDisparityBilateralFilter(filter_size, refine_iter);
    disp_filter_->setEdgeThreshold(edge_thresh);
    disp_filter_->setMaxDiscThreshold(disc_thresh);
    disp_filter_->setSigmaRange(sigma_range);
}

void StereoMatcher::computeDisparity(cv::cuda::GpuMat &d_left, cv::cuda::GpuMat &d_right, cv::cuda::GpuMat &d_disp)
{
    cv::cuda::GpuMat d_left_, d_right_;
    d_left_.create(d_left.size(), CV_8UC1);
    d_right_.create(d_right.size(), CV_8UC1);

    cv::cuda::cvtColor(d_left, d_left_, cv::COLOR_BGR2GRAY);
    cv::cuda::cvtColor(d_right, d_right_, cv::COLOR_BGR2GRAY);

    blur_filter_->apply(d_left_, d_left_);
    blur_filter_->apply(d_right_, d_right_);

    stereo_matcher_->compute(d_left_, d_right_, d_disp);
    if (use_filter_)
        disp_filter_->apply(d_disp, d_left, d_disp);
}

} // namespace cuda
