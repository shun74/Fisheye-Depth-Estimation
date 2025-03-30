#include "pcd_generator.h"

PointCloudGenerator::PointCloudGenerator(double fx, double fy, double cx, double cy, double base, const cv::Mat &map_x,
                                         const cv::Mat &map_y)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), base_(base), convert_map_x_(map_x.clone()), convert_map_y_(map_y.clone()),
      box_limits_set_(false)
{
}

void PointCloudGenerator::setBoxLimits(double x_min, double x_max, double y_min, double y_max, double z_min,
                                       double z_max)
{
    x_min_ = x_min;
    x_max_ = x_max;
    y_min_ = y_min;
    y_max_ = y_max;
    z_min_ = z_min;
    z_max_ = z_max;
    box_limits_set_ = true;
}

bool PointCloudGenerator::calcPoint(int px, int py, float d, cv::Point3f *point)
{
    if (px < 0 || py < 0 || px >= convert_map_x_.cols || py >= convert_map_x_.rows)
        return false;

    if (px - d <= 0.0f || d < 1.0f)
        return false;

    const float lx = convert_map_x_.at<float>(py, px);

    const int disp_x = static_cast<int>(px - d);
    if (disp_x < 0 || disp_x >= convert_map_x_.cols)
        return false;

    const float rx = convert_map_x_.at<float>(py, disp_x);
    const float adjusted_d = lx - rx;

    if (std::fabs(adjusted_d) < 1e-6)
        return false;

    const float cy_val = convert_map_y_.at<float>(py, px);

    double z = fx_ * base_ / adjusted_d;
    double x = z * (lx - cx_) / fx_;
    double y = z * (cy_val - cy_) / fy_;

    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        return false;

    if (box_limits_set_)
    {
        if (x <= x_min_ || x >= x_max_ || y <= y_min_ || y >= y_max_ || z <= z_min_ || z >= z_max_)
            return false;
    }

    point->x = static_cast<float>(x);
    point->y = static_cast<float>(y);
    point->z = static_cast<float>(z);

    return true;
}

void PointCloudGenerator::computePointCloud(const cv::Mat &disp, cv::Mat &pcd, std::vector<bool> &valid)
{
    int w = disp.size().width;
    int h = disp.size().height;

    cv::Point3f *pcd_ptr = pcd.ptr<cv::Point3f>();
    valid.resize(w * h, true);

#pragma omp parallel for
    for (int py = 0; py < h; py++)
    {
        for (int px = 0; px < w; px++)
        {
            float d = static_cast<float>(disp.at<short>(py, px)) / 16.0f;
            if (!calcPoint(px, py, d, &pcd_ptr[py * w + px]))
            {
                pcd_ptr[py * w + px].x = 0;
                pcd_ptr[py * w + px].y = 0;
                pcd_ptr[py * w + px].z = 0;
                valid[py * w + px] = false;
            }
        }
    }
}

void PointCloudGenerator::computePointCloud(const cv::Mat &image, const cv::Mat &disp, cv::Mat &pcd, cv::Mat &colors,
                                            std::vector<bool> &valid)
{
    if (image.size().height * image.size().width != disp.size().height * disp.size().width)
        return;

    int w = disp.size().width;
    int h = disp.size().height;

    pcd.create(1, h * w, CV_32FC3);
    colors.create(1, h * w, CV_8UC3);

    const cv::Vec3b *image_ptr = image.ptr<cv::Vec3b>();
    cv::Point3f *pcd_ptr = pcd.ptr<cv::Point3f>();
    cv::Vec3b *colors_ptr = colors.ptr<cv::Vec3b>();
    valid.resize(w * h, true);

#pragma omp parallel for
    for (int py = 0; py < h; py++)
    {
        for (int px = 0; px < w; px++)
        {
            float d = static_cast<float>(disp.at<short>(py, px)) / 16.0f;
            if (!calcPoint(px, py, d, &pcd_ptr[py * w + px]))
            {
                pcd_ptr[py * w + px].x = 0;
                pcd_ptr[py * w + px].y = 0;
                pcd_ptr[py * w + px].z = 0;
                valid[py * w + px] = false;
            }
            // BGR -> RGB
            colors_ptr[py * w + px][0] = image_ptr[py * w + px][2];
            colors_ptr[py * w + px][1] = image_ptr[py * w + px][1];
            colors_ptr[py * w + px][2] = image_ptr[py * w + px][0];
        }
    }
}
