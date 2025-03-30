#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

class StereoVisionProcessor
{
  public:
    virtual ~StereoVisionProcessor() = default;

    virtual void setPostFilter(float lambda, float sigma) = 0;

    virtual void setBoxLimits(double x_min, double x_max, double y_min, double y_max, double z_min, double z_max) = 0;

    virtual void computeDisparity(cv::InputArray _left, cv::InputArray _right, cv::OutputArray _disparity) = 0;

    virtual void computePointCloud(cv::InputArray _left, cv::InputArray _right, cv::InputOutputArray _disparity,
                                   cv::OutputArray _pcd, cv::OutputArray _colors, std::vector<bool> &valid) = 0;

    static std::unique_ptr<StereoVisionProcessor> create(bool gray_scale, std::string algorithm, cv::Size blur_kernel,
                                                         int min_disp, int max_disp, int block_size, int p1, int p2,
                                                         int max_diff, int pre_fc, int unique_ratio, int speckle_size,
                                                         int speckle_range, int mode, double fx, double fy, double cx,
                                                         double cy, double base, const cv::Mat &map_x,
                                                         const cv::Mat &map_y);
};
