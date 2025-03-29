#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

class PointCloudGenerator
{
  private:
    double fx_, fy_, cx_, cy_, base_;

    cv::Mat convert_map_x_;
    cv::Mat convert_map_y_;

    bool box_limits_set_;
    double x_min_, x_max_;
    double y_min_, y_max_;
    double z_min_, z_max_;

    bool calcPoint(int px, int py, float d, cv::Point3f *point);

  public:
    PointCloudGenerator(double fx, double fy, double cx, double cy, double base, const cv::Mat &map_x,
                        const cv::Mat &map_y);

    void setBoxLimits(double x_min, double x_max, double y_min, double y_max, double z_min, double z_max);

    void computePointCloud(const cv::Mat &disp, cv::OutputArray _pcd, std::vector<bool> &valid);

    void computePointCloud(const cv::Mat &image, const cv::Mat &disp, cv::OutputArray _pcd, cv::OutputArray _colors,
                           std::vector<bool> &valid);
};
