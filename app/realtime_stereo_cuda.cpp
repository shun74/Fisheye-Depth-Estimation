#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>

#include <yaml-cpp/yaml.h>

#include <opencv2/opencv.hpp>

#include <stereo_vision_cuda.h>
#include <undistort.h>

#include "pcd_visualizer.h"

#define WINDOW_W 1280
#define WINDOW_H 960

bool exit_flag = false;

void handle_signal(int signal)
{
    std::cout << "\n" << "Caught signal " << signal << std::endl;
    exit_flag = true;
}

void run(cuda::StereoVisionProcessor *sv_processor, const std::vector<cv::Mat> &left_maps,
         const std::vector<cv::Mat> &right_maps, const cv::Size &img_size)
{
    int w = img_size.width;
    int h = img_size.height;

    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video capture." << std::endl;
        return;
    }
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, w * 2);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, h);

    cv::namedWindow("input", cv::WINDOW_NORMAL);
    cv::resizeWindow("input", WINDOW_W, WINDOW_H);
    cv::namedWindow("eqrec", cv::WINDOW_NORMAL);
    cv::resizeWindow("eqrec", WINDOW_W, WINDOW_H);
    cv::namedWindow("disparity", cv::WINDOW_NORMAL);
    cv::resizeWindow("disparity", WINDOW_W, WINDOW_H);

    const int max_points = w * h;

    PointCloudVisualizer visualizer(WINDOW_W, WINDOW_H, max_points);
    visualizer.setPointSize(4);

    // visualizer.setDownsamplingFactor(2); // reduce point to half

    visualizer.start();

    cv::Mat img(h, w * 2, CV_8UC3);
    cv::Mat eqrec_left(h, w, CV_8UC3);
    cv::Mat eqrec_right(h, w, CV_8UC3);
    cv::Mat eqrec(h, w, CV_8UC3);
    cv::Mat raw_disp(h, w, CV_16S);
    cv::Mat norm_disp(h, w, CV_8UC1);
    cv::Mat point_cloud(1, w * h, CV_32FC3);
    cv::Mat colors(1, w * h, CV_8UC3);
    std::vector<bool> valid;

    float fps = 0.0f;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    double elapsed_time;

    while (!exit_flag)
    {
        start_time = std::chrono::high_resolution_clock::now();
        if (cv::waitKey(1) >= 0)
            exit_flag = true;

        cap >> img;
        if (img.empty())
            continue;

        cv::remap(img(cv::Rect(0, 0, w, h)), eqrec_left, left_maps[0], left_maps[1], cv::INTER_LINEAR,
                  cv::BORDER_CONSTANT);
        cv::remap(img(cv::Rect(w, 0, w, h)), eqrec_right, right_maps[0], right_maps[1], cv::INTER_LINEAR,
                  cv::BORDER_CONSTANT);
        sv_processor->computePointCloud(eqrec_left, eqrec_right, raw_disp, point_cloud, colors, valid);

        visualizer.updatePointCloud(point_cloud, colors, valid, fps);

        cv::imshow("input", img);
        cv::hconcat(eqrec_left, eqrec_right, eqrec);
        cv::imshow("eqrec", eqrec);
        cv::normalize(raw_disp, norm_disp, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imshow("disparity", norm_disp);
        end_time = std::chrono::high_resolution_clock::now();
        elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
        fps = 1.0 / elapsed_time;
    }

    visualizer.stop();
    cv::destroyAllWindows();
}

int main(int argc, char **argv)
{
    // monitor the signales to stop
    std::signal(SIGINT, handle_signal);

    // load config file
    std::string command_line_keys = "{config | configs/realtime_stereo_cuda.yaml | path to the config file |}"
                                    "{calib | configs/camera_params.yaml | path to the camera parameters file |}";
    cv::CommandLineParser parser(argc, argv, command_line_keys);
    std::string conf_path = parser.get<std::string>("config");
    std::string calib_path = parser.get<std::string>("calib");

    // Load config yaml file
    YAML::Node config = YAML::LoadFile(conf_path);
    YAML::Node sm_config = config["stereo_matcher"];
    YAML::Node pf_config = config["post_filter"];
    cv::Size blur_kernel = cv::Size(sm_config["blur_kernel"][0].as<int>(), sm_config["blur_kernel"][1].as<int>());
    int min_disp = sm_config["min_disp"].as<int>();
    int max_disp = sm_config["max_disp"].as<int>();
    int p1 = sm_config["p1"].as<int>();
    int p2 = sm_config["p2"].as<int>();
    int unique_ratio = sm_config["unique_ratio"].as<int>();
    int mode = sm_config["mode"].as<int>();
    int filter_size = pf_config["filter_size"].as<int>();
    int refine_iter = pf_config["refine_iter"].as<int>();
    double edge_thresh = pf_config["edge_thresh"].as<double>();
    double disc_thresh = pf_config["disc_thresh"].as<double>();
    double sigma_range = pf_config["sigma_range"].as<double>();

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

    // create conversion maps
    std::vector<cv::Mat> left_maps(2);
    std::vector<cv::Mat> right_maps(2);
    cv::Mat map_x, map_y;
    computeStereoRectifyMaps(K1, K2, D1, D2, R1, R2, P1, P2, img_size, left_maps, right_maps, CV_32F);
    computeEquirectangleMaps(P1, img_size, map_x, map_y);

    std::unique_ptr<cuda::StereoVisionProcessor> sv_processor = cuda::StereoVisionProcessor::create(
        blur_kernel, min_disp, max_disp, p1, p2, unique_ratio, mode, fx, fy, cx, cy, base_line, map_x, map_y);
    sv_processor->setPostFilter(filter_size, refine_iter, edge_thresh, disc_thresh, sigma_range);

    run(sv_processor.get(), left_maps, right_maps, img_size);

    return 0;
}
