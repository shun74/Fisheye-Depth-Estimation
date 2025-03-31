#include <opencv2/core/cuda.hpp>

#include <stereo_vision_cuda.h>

#include "pcd_generator_cuda.h"
#include "stereo_matcher_cuda.h"

class StereoVisionProcessorImpl : public cuda::StereoVisionProcessor
{
  public:
    StereoVisionProcessorImpl(cv::Size blur_kernel, int min_disp, int max_disp, int p1, int p2, int unique_ratio,
                              int mode, double fx, double fy, double cx, double cy, double base_line,
                              const cv::Mat &map_x, const cv::Mat &map_y)
        : stereo_matcher_(
              std::make_unique<cuda::StereoMatcher>(blur_kernel, min_disp, max_disp, p1, p2, unique_ratio, mode)),
          pcd_generator_(std::make_unique<cuda::PointCloudGenerator>(fx, fy, cx, cy, base_line, map_x, map_y))
    {
    }

    void setPostFilter(int filter_size, int refine_iter, double edge_thresh, double disc_thresh,
                       double sigma_range) override
    {
        stereo_matcher_->setPostFilter(filter_size, refine_iter, edge_thresh, disc_thresh, sigma_range);
    }

    void setBoxLimits(double x_min, double x_max, double y_min, double y_max, double z_min, double z_max) override
    {
        pcd_generator_->setBoxLimits(x_min, x_max, y_min, y_max, z_min, z_max);
    }

    void computeDisparity(cv::InputArray _left, cv::InputArray _right, cv::OutputArray _disparity) override
    {
        cv::Mat h_left = _left.getMat();
        cv::Mat h_right = _right.getMat();

        cv::cuda::GpuMat d_left, d_right, d_disparity;
        d_left.upload(h_left);
        d_right.upload(h_right);

        stereo_matcher_->computeDisparity(d_left, d_right, d_disparity);

        d_disparity.download(_disparity);
    }

    void computePointCloud(cv::InputArray _left, cv::InputArray _right, cv::InputOutputArray _disparity,
                           cv::OutputArray _pcd, cv::OutputArray _colors, std::vector<bool> &valid) override
    {
        int w = _left.size().width;
        int h = _left.size().height;

        cv::Mat h_left = _left.getMat();
        cv::Mat h_right = _right.getMat();

        cv::cuda::GpuMat d_left, d_right, d_disparity, d_disparity_float;
        d_left.upload(h_left);
        d_right.upload(h_right);

        stereo_matcher_->computeDisparity(d_left, d_right, d_disparity);

        d_disparity.convertTo(d_disparity_float, CV_32FC1);

        cv::cuda::GpuMat d_pcd;
        d_pcd.create(1, h * w, CV_32FC3);
        pcd_generator_->computePointCloud(d_disparity_float, d_pcd, valid);

        _disparity.create(d_disparity.rows, d_disparity.cols, d_disparity.type());
        d_disparity.download(_disparity);
        _pcd.create(d_pcd.size(), d_pcd.type());
        d_pcd.download(_pcd);

        _colors.create(1, h * w, CV_8UC3);
        cv::Mat h_colors = _colors.getMat();
        h_left.reshape(3, 1).copyTo(h_colors);
        cv::cvtColor(h_colors, h_colors, cv::COLOR_BGR2RGB);

        d_left.release();
        d_right.release();
        d_disparity.release();
        d_pcd.release();
    }

  private:
    std::unique_ptr<cuda::StereoMatcher> stereo_matcher_;
    std::unique_ptr<cuda::PointCloudGenerator> pcd_generator_;
};

namespace cuda
{

std::unique_ptr<StereoVisionProcessor> StereoVisionProcessor::create(cv::Size blur_kernel, int min_disp, int max_disp,
                                                                     int p1, int p2, int unique_ratio, int mode,
                                                                     double fx, double fy, double cx, double cy,
                                                                     double base_line, const cv::Mat &map_x,
                                                                     const cv::Mat &map_y)
{
    return std::make_unique<StereoVisionProcessorImpl>(blur_kernel, min_disp, max_disp, p1, p2, unique_ratio, mode, fx,
                                                       fy, cx, cy, base_line, map_x, map_y);
}

} // namespace cuda