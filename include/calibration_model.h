#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

void loadStereoImages(std::string img_dir, std::vector<cv::Mat> &left_imgs, std::vector<cv::Mat> &right_imgs);

class CalibrationModel
{
  public:
    virtual ~CalibrationModel() = default;

    virtual void calibrate(std::vector<cv::Mat> &left_imgs, std::vector<cv::Mat> &right_imgs) = 0;

    virtual void writeCameraParams(std::string path) = 0;

    static std::unique_ptr<CalibrationModel> create(cv::Size img_size, cv::Size board_size, float square_size, float fov);
};
