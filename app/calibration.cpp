#include <iostream>

#include <yaml-cpp/yaml.h>

#include <opencv2/opencv.hpp>

#include <calibration_model.h>
#include <undistort.h>

#define WINDOW_W 1280
#define WINDOW_H 960

int main(int argc, char **argv)
{

    // load config file
    std::string command_line_keys = "{config | configs/calibration.yaml | path to the config file |}"
                                    "{images_dir | calib_images/ | path to the calibration images directory |}"
                                    "{output | configs/camera_params.yaml | path to save the camera parameters |}"
                                    "{test_image | images/test-1.jpg | path to the test image |}";
    cv::CommandLineParser parser(argc, argv, command_line_keys);
    std::string conf_path = parser.get<std::string>("config");
    std::string imgs_dir = parser.get<std::string>("images_dir");
    std::string save_path = parser.get<std::string>("output");
    std::string test_path = parser.get<std::string>("test_image");

    YAML::Node config = YAML::LoadFile(conf_path);
    cv::Size img_size(config["image_size"][0].as<int>(), config["image_size"][1].as<int>());
    cv::Size board_size(config["board_size"][0].as<int>(), config["board_size"][1].as<int>());
    float square_size = config["square_size"].as<float>();
    float fov = config["fov"].as<float>();

    std::unique_ptr<CalibrationModel> calib_model = CalibrationModel::create(img_size, board_size, square_size, fov);

    std::vector<cv::Mat> left_imgs, right_imgs;
    loadStereoImages(imgs_dir, left_imgs, right_imgs);

    calib_model->calibrate(left_imgs, right_imgs);
    calib_model->writeCameraParams(save_path);

    cv::FileStorage fs(save_path, cv::FileStorage::READ);

    cv::FileNode cp = fs["camera_params"];
    cv::Mat K1, K2, D1, D2, R1, R2, P1, P2;
    cp["K1"] >> K1;
    cp["K2"] >> K2;
    cp["D1"] >> D1;
    cp["D2"] >> D2;
    cp["R1"] >> R1;
    cp["R2"] >> R2;
    cp["P1"] >> P1;
    cp["P2"] >> P2;

    // test sample image
    std::vector<cv::Mat> left_maps(2);
    std::vector<cv::Mat> right_maps(2);
    computeStereoRectifyMaps(K1, K2, D1, D2, R1, R2, P1, P2, img_size, left_maps, right_maps, CV_32F);

    cv::Mat test_img = cv::imread(test_path);
    int w = test_img.cols / 2;
    int h = test_img.rows;
    cv::Mat undistorted_img, undistorted_left, undistorted_right;

    cv::remap(test_img(cv::Rect(0, 0, w, h)), undistorted_left, left_maps[0], left_maps[1], cv::INTER_LINEAR,
              cv::BORDER_CONSTANT);
    cv::remap(test_img(cv::Rect(w, 0, w, h)), undistorted_right, right_maps[0], right_maps[1], cv::INTER_LINEAR,
              cv::BORDER_CONSTANT);
    cv::hconcat(undistorted_left, undistorted_right, undistorted_img);
    cv::line(undistorted_img, cv::Point(0, h / 2), cv::Point(w * 2, h / 2), cv::Scalar(0, 255, 0), 2);

    cv::namedWindow("input", cv::WINDOW_NORMAL);
    cv::resizeWindow("input", WINDOW_W, WINDOW_H);
    cv::imshow("input", test_img);

    cv::namedWindow("undistorted", cv::WINDOW_NORMAL);
    cv::resizeWindow("undistorted", WINDOW_W, WINDOW_H);
    cv::imshow("undistorted", undistorted_img);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
