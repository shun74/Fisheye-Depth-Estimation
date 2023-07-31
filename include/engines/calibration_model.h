#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ccalib/omnidir.hpp>
#include <configs/camera_type.h>
#include <configs/calibration_model_params.h>

namespace engine
{

void stereoImageLoader(std::string img_dir, std::vector<cv::Mat> &left_imgs, std::vector<cv::Mat> &right_imgs);

class CalibrationModel
{
  private:
    // Base
    CameraType cam_type_;
    cv::Size img_size_;
    cv::Size board_size_;
    float square_size_;

    // flag
    bool is_calibrated_;

    // Camera
    cv::Mat K1_, K2_, D1_, D2_, R1_, R2_, P1_, P2_, R_, T_;

    // fisheye & omnidir
    float fov_;
    cv::Mat xi1_, xi2_;

    void scanCheckerBoard(
        std::vector<cv::Mat> &left_imgs, std::vector<cv::Mat> &right_imgs,
        std::vector<std::vector<cv::Point2f>> &left_corners,
        std::vector<std::vector<cv::Point2f>> &right_corners);

    void getObjPoints(int num, std::vector<std::vector<cv::Point3f>> &obj_points);

    void calibratePinhole(
        std::vector<std::vector<cv::Point2f>> left_points,
        std::vector<std::vector<cv::Point2f>> right_points,
        std::vector<std::vector<cv::Point3f>> obj_points);
    void calibrateFisheye(
        std::vector<std::vector<cv::Point2f>> left_points,
        std::vector<std::vector<cv::Point2f>> right_points,
        std::vector<std::vector<cv::Point3f>> obj_points);
    void calibrateOmnidir(
        std::vector<std::vector<cv::Point2f>> left_points,
        std::vector<std::vector<cv::Point2f>> right_points,
        std::vector<std::vector<cv::Point3f>> obj_points);

  public:
    CalibrationModel(std::string path);
    CalibrationModel(config::CalibrationModelParams cm_params);

    void setParams(config::CalibrationModelParams cm_params);

    void calibrate(std::vector<cv::Mat> &left_imgs, std::vector<cv::Mat> &right_imgs);

    void writeCameraParams(std::string path);
};

}