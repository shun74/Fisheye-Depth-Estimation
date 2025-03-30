#include "stereo_matcher.h"

StereoMatcher::StereoMatcher(bool gray_scale, std::string algorithm, cv::Size blur_kernel, int max_disp, int block_size)
    : gray_scale_(gray_scale), algorithm_(algorithm), blur_kernel_(blur_kernel), use_filter_(false)
{
    stereo_matcher_ = cv::StereoBM::create(max_disp, block_size);
}

StereoMatcher::StereoMatcher(bool gray_scale, std::string algorithm, cv::Size blur_kernel, int min_disp, int max_disp,
                             int block_size, int p1, int p2, int max_diff, int pre_fc, int unique_ratio,
                             int speckle_size, int speckle_range, int mode)
    : gray_scale_(gray_scale), algorithm_(algorithm), blur_kernel_(blur_kernel), use_filter_(false)
{
    stereo_matcher_ = cv::StereoSGBM::create(min_disp, max_disp, block_size, p1, p2, max_diff, pre_fc, unique_ratio,
                                             speckle_size, speckle_range, mode);
}

void StereoMatcher::setPostFilter(float lambda, float sigma)
{
    use_filter_ = true;
    wls_filter_ = cv::ximgproc::createDisparityWLSFilter(stereo_matcher_);
    stereo_matcher_r_ = cv::ximgproc::createRightMatcher(stereo_matcher_);
    wls_filter_->setLambda(lambda);
    wls_filter_->setSigmaColor(sigma);
}

void StereoMatcher::computeDisparity(const cv::Mat &left, const cv::Mat &right, cv::Mat &disp)
{
    disp.create(left.size(), CV_16S);
    
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
