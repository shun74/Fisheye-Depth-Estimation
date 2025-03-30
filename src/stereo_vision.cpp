#include <stereo_vision.h>

#include "pcd_generator.h"
#include "stereo_matcher.h"

class StereoVisionProcessorImpl : public StereoVisionProcessor
{
  public:
    StereoVisionProcessorImpl(bool gray_scale, std::string algorithm, cv::Size blur_kernel, int min_disp, int max_disp,
                              int block_size, int p1, int p2, int max_diff, int pre_fc, int unique_ratio,
                              int speckle_size, int speckle_range, int mode, double fx, double fy, double cx, double cy,
                              double base, const cv::Mat &map_x, const cv::Mat &map_y)
        : stereo_matcher_(std::make_unique<StereoMatcher>(gray_scale, algorithm, blur_kernel, min_disp, max_disp,
                                                          block_size, p1, p2, max_diff, pre_fc, unique_ratio,
                                                          speckle_size, speckle_range, mode)),
          pcd_generator_(std::make_unique<PointCloudGenerator>(fx, fy, cx, cy, base, map_x, map_y))
    {
    }

    void setPostFilter(float lambda, float sigma) override
    {
        stereo_matcher_->setPostFilter(lambda, sigma);
    }

    void setBoxLimits(double x_min, double x_max, double y_min, double y_max, double z_min, double z_max) override
    {
        pcd_generator_->setBoxLimits(x_min, x_max, y_min, y_max, z_min, z_max);
    }

    void computeDisparity(cv::InputArray _left, cv::InputArray _right, cv::OutputArray _disparity) override
    {
        cv::Mat left = _left.getMat();
        cv::Mat right = _right.getMat();
        cv::Mat disparity = _disparity.getMat();

        stereo_matcher_->computeDisparity(left, right, disparity);
    }

    void computePointCloud(cv::InputArray _left, cv::InputArray _right, cv::InputOutputArray _disparity,
                           cv::OutputArray _pcd, cv::OutputArray _colors, std::vector<bool> &valid) override
    {
        cv::Mat left = _left.getMat();
        cv::Mat right = _right.getMat();
        cv::Mat disparity = _disparity.getMat();
        cv::Mat pcd = _pcd.getMat();
        cv::Mat colors = _colors.getMat();

        stereo_matcher_->computeDisparity(left, right, disparity);
        pcd_generator_->computePointCloud(left, disparity, pcd, colors, valid);
    }

  private:
    std::unique_ptr<StereoMatcher> stereo_matcher_;
    std::unique_ptr<PointCloudGenerator> pcd_generator_;
};

std::unique_ptr<StereoVisionProcessor> StereoVisionProcessor::create(
    bool gray_scale, std::string algorithm, cv::Size blur_kernel, int min_disp, int max_disp, int block_size, int p1,
    int p2, int max_diff, int pre_fc, int unique_ratio, int speckle_size, int speckle_range, int mode, double fx,
    double fy, double cx, double cy, double base, const cv::Mat &map_x, const cv::Mat &map_y)
{
    return std::make_unique<StereoVisionProcessorImpl>(gray_scale, algorithm, blur_kernel, min_disp, max_disp,
                                                       block_size, p1, p2, max_diff, pre_fc, unique_ratio, speckle_size,
                                                       speckle_range, mode, fx, fy, cx, cy, base, map_x, map_y);
}
