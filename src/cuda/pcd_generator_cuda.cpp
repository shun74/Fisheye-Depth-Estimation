#include "pcd_generator_cuda.h"
#include "calculate_points.h"

namespace cuda
{

PointCloudGenerator::PointCloudGenerator(double fx, double fy, double cx, double cy, double base, const cv::Mat &map_x,
                                         const cv::Mat &map_y)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), base_(base), box_limits_set_(false)
{
    convert_map_x_.upload(map_x);
    convert_map_y_.upload(map_y);
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

void PointCloudGenerator::computePointCloud(const cv::cuda::GpuMat &d_disp, cv::cuda::GpuMat &d_pcd,
                                            std::vector<bool> &h_valid)
{
    int w = d_disp.cols;
    int h = d_disp.rows;

    if (d_pcd.empty() || d_pcd.size() != cv::Size(w, h) || d_pcd.type() != CV_32FC3)
        d_pcd.create(1, h * w, CV_32FC3);

    bool *d_valid;
    cudaMalloc((void **)&d_valid, sizeof(bool) * w * h);
    cudaMemset(d_valid, 0, sizeof(bool) * w * h);

    calc_points(d_disp.ptr<float>(), convert_map_x_.ptr<float>(), convert_map_y_.ptr<float>(), w, h,
                static_cast<float>(fx_), static_cast<float>(fy_), static_cast<float>(cx_), static_cast<float>(cy_),
                static_cast<float>(base_), d_pcd.ptr<float>(), d_valid);

    bool *h_valid_buffer = new bool[w * h];
    cudaMemcpy(h_valid_buffer, d_valid, sizeof(bool) * w * h, cudaMemcpyDeviceToHost);

    h_valid.resize(w * h);
    for (int i = 0; i < w * h; i++)
    {
        h_valid[i] = h_valid_buffer[i];
    }
    delete[] h_valid_buffer;
    cudaFree(d_valid);

    // please implement this function if u need
    // if (box_limits_set_)
    // {
    //     filterPointsOutsideBox(pcd, valid);
    // }
}

} // namespace cuda
