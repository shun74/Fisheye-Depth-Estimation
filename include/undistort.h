#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

void computeStereoRectifyMaps(cv::Mat K1, cv::Mat K2, cv::Mat D1, cv::Mat D2, cv::Mat R1, cv::Mat R2, cv::Mat P1,
                              cv::Mat P2, cv::Size img_size, std::vector<cv::Mat> &left_maps,
                              std::vector<cv::Mat> &right_maps, int dtype = CV_32F);

void computeEquirectangleMaps(cv::Mat K, cv::Size img_size, cv::Mat &eqrec_map_x, cv::Mat &eqrec_map_y);
