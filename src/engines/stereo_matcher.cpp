#include <engines/stereo_matcher.h>

namespace engine
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
    gray_scale_ = sm_p.gray_scale;
    algorithm_ = sm_p.algorithm;
    blur_kernel_ = sm_p.blur_kernel;

    if (algorithm_ == "StereoBM")
    {
        stereo_matcher_ = cv::StereoBM::create(sm_p.max_disp, sm_p.block_size);
    }
    else if (algorithm_ == "StereoSGBM")
    {
        stereo_matcher_ = cv::StereoSGBM::create(
            sm_p.min_disp, sm_p.max_disp,
            sm_p.block_size, sm_p.p1, sm_p.p2,
            sm_p.max_diff, sm_p.pre_fc, sm_p.unique_ratio,
            sm_p.speckle_size, sm_p.speckle_range, sm_p.mode);
    }

    use_filter_ = sm_p.use_filter;
    if (use_filter_)
    {
        wls_filter_ = cv::ximgproc::createDisparityWLSFilter(stereo_matcher_);
        stereo_matcher_r_ = cv::ximgproc::createRightMatcher(stereo_matcher_);
        wls_filter_->setLambda(sm_p.wsl_lambda);
        wls_filter_->setSigmaColor(sm_p.wsl_sigma);
    }
}

void StereoMatcher::computeDisparity(cv::Mat &left, cv::Mat &right, cv::Mat &disp)
{
    if (gray_scale_)
    {
        cv::cvtColor(left, left, cv::COLOR_BGR2GRAY);
        cv::cvtColor(right, right, cv::COLOR_BGR2GRAY);
    }
    
    cv::blur(left, left, blur_kernel_);
    cv::blur(right, right, blur_kernel_);

    if (use_filter_)
    {
        cv::Mat dispL, dispR;
        stereo_matcher_->compute(left, right, dispL);
        stereo_matcher_r_->compute(right, left, dispR);
        wls_filter_->filter(dispL, left, disp, dispR);
    }
    else
    {
        stereo_matcher_->compute(left, right, disp);
    }
}

} // namespace engine