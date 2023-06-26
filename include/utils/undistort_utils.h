#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/ccalib/omnidir.hpp>
#include <configs/camera_type.h>

namespace undistort_utils
{

void computeStereoRectifyMaps(
    cv::Mat K1, cv::Mat K2, cv::Mat D1, cv::Mat D2,
    cv::Mat R1, cv::Mat R2, cv::Mat P1, cv::Mat P2,
    cv::Size img_size, CameraType camera,
    std::vector<cv::Mat> &left_maps, std::vector<cv::Mat> &right_maps,
    int dtype=CV_32F, cv::Mat xi1=cv::Mat::zeros(1,1,CV_32F), cv::Mat xi2=cv::Mat::zeros(1,1,CV_32F), int omni_type=cv::omnidir::RECTIFY_LONGLATI);

void computeFisheyeMap(
    cv::Mat K, cv::Mat D, cv::Mat R, cv::Mat P,
    cv::Size img_size, int dtype, 
    cv::Mat &map_x, cv::Mat &map_y);

void computeEquirectangleMaps(
    cv::Mat K, cv::Size img_size,
    cv::Mat &eqrec_map_x,
    cv::Mat &eqrec_map_y);

}
