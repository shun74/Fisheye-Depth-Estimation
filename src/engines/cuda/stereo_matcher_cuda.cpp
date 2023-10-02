#include <engines/cuda/stereo_matcher_cuda.h>

namespace engine
{
namespace cuda
{

StereoMatcher::StereoMatcher(std::string path)
{
  config::StereoMatcherParams sm_params(path);
  setParameters(sm_params);
}

StereoMatcher::StereoMatcher(config::StereoMatcherParams sm_params)
{
  setParameters(sm_params);
}

void StereoMatcher::setParameters(config::StereoMatcherParams sm_p)
{
  algorithm_ = sm_p.algorithm;
  blur_kernel_ = sm_p.blur_kernel;

  stereo_matcher_ = cv::cuda::createStereoSGM(
      sm_p.min_disp, sm_p.max_disp, sm_p.p1, sm_p.p2,
      sm_p.unique_ratio, sm_p.mode);

  use_filter_ = sm_p.use_filter;
  if (use_filter_) {
    disp_filter_ = cv::cuda::createDisparityBilateralFilter(sm_p.max_disp, sm_p.dbf_filter_size, sm_p.dbf_refine_iter);
    disp_filter_->setEdgeThreshold(sm_p.dbf_edge_thresh);
    disp_filter_->setMaxDiscThreshold(sm_p.dbf_disc_thresh);
    disp_filter_->setSigmaRange(sm_p.dbf_sigma_range);
  }
}

void StereoMatcher::computeDisparity(cv::Mat &left, cv::Mat &right, cv::Mat &disp)
{
    
  cv::cuda::GpuMat d_color, d_left, d_right, d_disp;
  d_color.upload(left);
  cv::blur(left, left, blur_kernel_);
  cv::blur(right, right, blur_kernel_);
  cv::cvtColor(left, left, cv::COLOR_BGR2GRAY);
  cv::cvtColor(right, right, cv::COLOR_BGR2GRAY);

  d_left.upload(left);
  d_right.upload(right);

  stereo_matcher_->compute(d_left, d_right, d_disp);
  if (use_filter_) disp_filter_->apply(d_disp, d_color, d_disp);

  d_disp.download(disp);
}

} // namespace cuda
} // namespace engine

