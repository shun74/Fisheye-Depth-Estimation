#pragma once

#include <vector>

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

namespace cuda
{

class PointCloudGenerator
{
  private:
    double fx_, fy_, cx_, cy_, base_;

    cv::cuda::GpuMat convert_map_x_;
    cv::cuda::GpuMat convert_map_y_;

    bool box_limits_set_;
    double x_min_, x_max_;
    double y_min_, y_max_;
    double z_min_, z_max_;

  public:
    PointCloudGenerator(double fx, double fy, double cx, double cy, double base, const cv::Mat &map_x,
                        const cv::Mat &map_y);

    void setBoxLimits(double x_min, double x_max, double y_min, double y_max, double z_min, double z_max);

    void computePointCloud(const cv::cuda::GpuMat &d_disp, cv::cuda::GpuMat &d_pcd, std::vector<bool> &h_valid);
};

} // namespace cuda
