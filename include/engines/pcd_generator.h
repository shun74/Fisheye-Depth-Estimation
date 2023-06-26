#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <configs/pcd_generator_params.h>

namespace engine
{

class PointCloudGenerator
{
    private:
        bool is_fisheye_;

        // fisheye
        bool set_maps_;
        cv::Mat convert_map_x_;
        cv::Mat convert_map_y_;

        double x_min_, x_max_;
        double y_min_, y_max_;
        double z_min_, z_max_;

        // camera
        double fx_, fy_, cx_, cy_, base_;

        bool calcPoint(int px, int py, float d, pcl::PointXYZ &point);

        bool calcColorPoint(int px, int py, float d, cv::Vec3b bgr, pcl::PointXYZRGB &point);

    public:
        PointCloudGenerator(std::string path);
        PointCloudGenerator(config::PointCloudGeneratorParams pcd_params);

        void setParams(config::PointCloudGeneratorParams pcd_params);

        void setConvertMaps(cv::Mat convert_map_x, cv::Mat convert_map_y);

        void computePointCloud(const cv::Mat &disp, pcl::PointCloud<pcl::PointXYZ>::Ptr &pcd);
        
        void computePointCloud(const cv::Mat &color, const cv::Mat &disp, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcd);
};

}