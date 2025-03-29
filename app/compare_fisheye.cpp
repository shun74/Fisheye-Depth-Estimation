#include <iostream>

#include <yaml-cpp/yaml.h>

#include <opencv2/opencv.hpp>

#include <stereo_vision.h>
#include <undistort.h>

#define WINDOW_W 1280
#define WINDOW_H 960

int main(int argc, char **argv)
{
    // load config file
    std::string command_line_keys = "{config | configs/single_test.yaml | path to the config file |}"
                                    "{calib | configs/camera_params.yaml | path to the camera parameters file |}"
                                    "{test_image | images/test-1.jpg | path to the test image |}";
    cv::CommandLineParser parser(argc, argv, command_line_keys);
    std::string conf_path = parser.get<std::string>("config");
    std::string calib_path = parser.get<std::string>("calib");
    std::string img_path = parser.get<std::string>("test_image");

    // Load config yaml file
    YAML::Node config = YAML::LoadFile(conf_path);
    YAML::Node sm_config = config["stereo_matcher"];
    YAML::Node pf_config = config["post_filter"];
    bool gray_scale = sm_config["gray_scale"].as<bool>();
    std::string algorithm = sm_config["algorithm"].as<std::string>();
    cv::Size blur_kernel = cv::Size(sm_config["blur_kernel"][0].as<int>(), sm_config["blur_kernel"][1].as<int>());
    int block_size = sm_config["block_size"].as<int>();
    int min_disp = sm_config["min_disp"].as<int>();
    int max_disp = sm_config["max_disp"].as<int>();
    int p1 = sm_config["p1"].as<int>();
    int p2 = sm_config["p2"].as<int>();
    int max_diff = sm_config["max_diff"].as<int>();
    int pre_fc = sm_config["pre_fc"].as<int>();
    int unique_ratio = sm_config["unique_ratio"].as<int>();
    int speckle_size = sm_config["speckle_size"].as<int>();
    int speckle_range = sm_config["speckle_range"].as<int>();
    int mode = sm_config["mode"].as<int>();
    double wsl_lambda = pf_config["wsl_lambda"].as<double>();
    double wsl_sigma = pf_config["wsl_sigma"].as<double>();

    // Load camera parameters yaml file
    cv::FileStorage fs(calib_path, cv::FileStorage::READ);
    cv::FileNode cp = fs["camera_params"];
    cv::Size img_size;
    fs["img_size"] >> img_size;
    cv::Mat K1, K2, D1, D2, R1, R2, P1, P2, T;
    cp["K1"] >> K1;
    cp["K2"] >> K2;
    cp["D1"] >> D1;
    cp["D2"] >> D2;
    cp["R1"] >> R1;
    cp["R2"] >> R2;
    cp["P1"] >> P1;
    cp["P2"] >> P2;
    cp["T"] >> T;
    double base_line = -T.at<double>(0);
    double fx = P1.at<double>(0, 0);
    double fy = P1.at<double>(1, 1);
    double cx = P1.at<double>(0, 2);
    double cy = P1.at<double>(1, 2);

    std::vector<cv::Mat> rect_left_maps(2);
    std::vector<cv::Mat> rect_right_maps(2);
    std::vector<cv::Mat> eqrec_left_maps(2);
    std::vector<cv::Mat> eqrec_right_maps(2);

    cv::fisheye::initUndistortRectifyMap(K1, D1, R1, P1, img_size, CV_32F, rect_left_maps[0], rect_left_maps[1]);
    cv::fisheye::initUndistortRectifyMap(K2, D2, R2, P2, img_size, CV_32F, rect_right_maps[0], rect_right_maps[1]);

    computeStereoRectifyMaps(K1, K2, D1, D2, R1, R2, P1, P2, img_size, eqrec_left_maps, eqrec_right_maps);
    cv::Mat eqrec_to_rect_map_x, eqrec_to_rect_map_y;
    computeEquirectangleMaps(P1, img_size, eqrec_to_rect_map_x, eqrec_to_rect_map_y);

    cv::Mat map_x, map_y;
    computeEquirectangleMaps(P1, img_size, map_x, map_y);
    std::unique_ptr<StereoVisionProcessor> sv_processor = StereoVisionProcessor::create(
        gray_scale, algorithm, blur_kernel, min_disp, max_disp, block_size, p1, p2, max_diff, pre_fc, unique_ratio,
        speckle_size, speckle_range, mode, fx, fy, cx, cy, base_line, map_x, map_y);
    sv_processor->setPostFilter(wsl_lambda, wsl_sigma);

    // test sample
    cv::Mat test_img = cv::imread(img_path);
    int w = test_img.cols / 2;
    int h = test_img.rows;

    cv::Mat rect_left, rect_right, eqrec_left, eqrec_right, disp, rect_disp_norm, eqrec_disp_norm;

    rect_left = test_img(cv::Rect(0, 0, w, h)).clone();
    rect_right = test_img(cv::Rect(w, 0, w, h)).clone();
    cv::remap(rect_left, rect_left, rect_left_maps[0], rect_left_maps[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::remap(rect_right, rect_right, rect_right_maps[0], rect_right_maps[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    sv_processor->computeDisparity(rect_left, rect_right, disp);

    eqrec_left = test_img(cv::Rect(0, 0, w, h)).clone();
    eqrec_right = test_img(cv::Rect(w, 0, w, h)).clone();
    cv::remap(eqrec_left, eqrec_left, eqrec_left_maps[0], eqrec_left_maps[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::remap(eqrec_right, eqrec_right, eqrec_right_maps[0], eqrec_right_maps[1], cv::INTER_LINEAR,
              cv::BORDER_CONSTANT);
    sv_processor->computeDisparity(eqrec_left, eqrec_right, disp);

    cv::namedWindow("input", cv::WINDOW_NORMAL);
    cv::resizeWindow("input", WINDOW_W, WINDOW_H);
    cv::imshow("input", test_img);

    cv::namedWindow("rectified", cv::WINDOW_NORMAL);
    cv::resizeWindow("rectified", WINDOW_W, WINDOW_H);
    cv::imshow("rectified", rect_left);

    cv::namedWindow("equirectangular", cv::WINDOW_NORMAL);
    cv::resizeWindow("equirectangular", WINDOW_W, WINDOW_H);
    cv::imshow("equirectangular", eqrec_left);

    cv::normalize(disp, rect_disp_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::namedWindow("rect_disp", cv::WINDOW_NORMAL);
    cv::resizeWindow("rect_disp", WINDOW_W, WINDOW_H);
    cv::imshow("rect_disp", rect_disp_norm);

    cv::normalize(disp, eqrec_disp_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::namedWindow("eqrec_disp", cv::WINDOW_NORMAL);
    cv::resizeWindow("eqrec_disp", WINDOW_W, WINDOW_H);
    cv::imshow("eqrec_disp", eqrec_disp_norm);

    cv::waitKey(0);
    cv::destroyAllWindows();

    cv::imwrite("output/rect.png", rect_left);
    cv::imwrite("output/rect_disp.png", rect_disp_norm);
    cv::imwrite("output/eqrec.png", eqrec_left);
    cv::imwrite("output/eqrec_disp.png", eqrec_disp_norm);

    return 0;
}
