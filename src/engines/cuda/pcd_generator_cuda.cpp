#include <engines/cuda/pcd_generator_cuda.h>

namespace engine
{

namespace cuda
{

PointCloudGenerator::PointCloudGenerator(std::string path)
{
    config::PointCloudGeneratorParams pcd_params(path);

    setParams(pcd_params);
    set_maps_ = false;
}

PointCloudGenerator::PointCloudGenerator(config::PointCloudGeneratorParams pcd_params)
{
    setParams(pcd_params);
    set_maps_ = false;
}

bool PointCloudGenerator::calcPoint(int px, int py, float d, pcl::PointXYZ &point)
{
    double x, y, z;
    if (is_fisheye_)
    {
        if (px-d<=0.0f || d<1.0f) return false;
        else
        {
            float lx = convert_map_x_.at<float>(py, px);
            float rx = convert_map_x_.at<float>(py, px - d);
            d = lx - rx;
        }
        z = fx_ * base_/d;
        x = z * (convert_map_x_.at<float>(py, px)-cx_)/fx_;
        y = z * (convert_map_y_.at<float>(py, px)-cy_)/fy_;
    }
    else
    {
        z = fx_ * base_/d;
        x = z * (px-cx_)/fx_;
        y = z * (py-cy_)/fy_;
    }
    if (!(x_min_<x&&x<x_max_&&y_min_<y&&y<y_max_&&z_min_<z&&z<z_max_)) return false;
    point.x = x;
    point.y = y;
    point.z = z;
    return true;
}

bool PointCloudGenerator::calcColorPoint(int px, int py, float d, cv::Vec3b bgr, pcl::PointXYZRGB &point)
{
    double x, y, z;

    if (is_fisheye_)
    {
        if (px-d<=0.0f || d<1.0f) return false;
        else
        {
            float lx = convert_map_x_.at<float>(py, px);
            float rx = convert_map_x_.at<float>(py, px - d);
            d = lx - rx;
        }
        z = fx_ * base_/d;
        x = z * (convert_map_x_.at<float>(py, px)-cx_)/fx_;
        y = z * (convert_map_y_.at<float>(py, px)-cy_)/fy_;
    }
    else
    {
        z = fx_ * base_/d;
        x = z * (px-cx_)/fx_;
        y = z * (py-cy_)/fy_;
    }
    if (!(x_min_<x&&x<x_max_&&y_min_<y&&y<y_max_&&z_min_<z&&z<z_max_)) return false;
    point.x = x;
    point.y = y;
    point.z = z;

    uint32_t rgb = ((uint32_t)bgr[2] << 16 | (uint32_t)bgr[1] << 8 | (uint32_t)bgr[0]);
    point.rgb = *reinterpret_cast<float *>(&rgb);
    return true;
}

void PointCloudGenerator::setParams(config::PointCloudGeneratorParams pcd_params)
{
    is_fisheye_ = pcd_params.is_fisheye;
    x_min_ = pcd_params.x_min;
    x_max_ = pcd_params.x_max;
    y_min_ = pcd_params.y_min;
    y_max_ = pcd_params.y_max;
    z_min_ = pcd_params.z_min;
    z_max_ = pcd_params.z_max;
    fx_ = pcd_params.fx;
    fy_ = pcd_params.fy;
    cx_ = pcd_params.cx;
    cy_ = pcd_params.cy;
    base_ = pcd_params.base;
    use_down_sample_ = pcd_params.use_down_sample;
    if (use_down_sample_) {
        leaf_x_ = pcd_params.leaf_x;
        leaf_y_ = pcd_params.leaf_y;
        leaf_z_ = pcd_params.leaf_z;
    }
}

void PointCloudGenerator::setConvertMaps(cv::Mat convert_map_x, cv::Mat convert_map_y)
{
    convert_map_x_ = convert_map_x;
    convert_map_y_ = convert_map_y;
    set_maps_ = true;
}

void PointCloudGenerator::computePointCloud(const cv::Mat &disp, pcl::PointCloud<pcl::PointXYZ>::Ptr &pcd)
{
    if (is_fisheye_ && !set_maps_)
    {
        std::cout << "Point Cloud Generator: Set maps befor run" << std::endl;
        return;
    }

    pcd->points.clear();
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> pcd_array(disp.size().height);
    cv::parallel_for_(cv::Range(0, disp.size().height), [&](const cv::Range& range) {
        for (int py=range.start; py<range.end; py++) {
            pcd_array[py] = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            for (int px=0; px<disp.size().width; px++) {
                float d = static_cast<float>(disp.at<short>(py, px)) / 16.0f;
                pcl::PointXYZ point;
                if (!calcPoint(px, py, d, point)) continue;
                pcd_array[py]->points.push_back(point);
            }
        }
    });

    for (auto &_pcd: pcd_array) {
        *pcd += *_pcd;
    }
    pcd->width = pcd->points.size();
    pcd->height = 1;
    pcd->is_dense = false;

    downSamplePoints<pcl::PointXYZ>(pcd);
}

void PointCloudGenerator::computePointCloud(const cv::Mat &color, const cv::Mat &disp, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcd)
{
    if (is_fisheye_ && !set_maps_)
    {
        std::cout << "Point Cloud Generator: Set maps befor run" << std::endl;
        return;
    }
    if (color.size().height*color.size().width != disp.size().height*disp.size().width)
        return;
    pcd->points.clear();

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcd_array(disp.size().height);
    cv::parallel_for_(cv::Range(0, disp.size().height), [&](const cv::Range& range) {
        for (int py=range.start; py<range.end; py++) {
            pcd_array[py] = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
            for (int px=0; px<disp.size().width; px++) {
                float d = static_cast<float>(disp.at<short>(py, px)) / 16.0f;
                cv::Vec3b bgr = color.at<cv::Vec3b>(py, px);
                pcl::PointXYZRGB point;
                if (!calcColorPoint(px, py, d, bgr, point)) continue;
                pcd_array[py]->points.push_back(point);
            }
        }
    });

    for (auto &_pcd: pcd_array) {
        *pcd += *_pcd;
    }

    pcd->width = pcd->points.size();
    pcd->height = 1;
    pcd->is_dense = false;

    downSamplePoints<pcl::PointXYZRGB>(pcd);
}

} // namespace cuda

} // namespace engine