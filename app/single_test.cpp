#include <iostream>

#include <yaml-cpp/yaml.h>

#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/viz/viz3d.hpp>

#include <stereo_vision.h>
#include <undistort.h>

#define WINDOW_W 1280
#define WINDOW_H 960

bool savePointCloudPLY(const cv::Mat &points, const cv::Mat &colors, const std::vector<bool> &valid,
                       const std::string &filename)
{
    try
    {
        std::vector<cv::Vec3f> validPoints;
        std::vector<cv::Vec3b> validColors;

        const float *point_ptr = points.ptr<float>(0);
        const uchar *color_ptr = colors.empty() ? nullptr : colors.ptr<uchar>(0);

        for (int i = 0; i < static_cast<int>(points.total()); i++)
        {
            if (valid.empty() || valid[i])
            {
                validPoints.push_back(cv::Vec3f(point_ptr[0], point_ptr[1], point_ptr[2]));

                if (color_ptr)
                {
                    validColors.push_back(cv::Vec3b(color_ptr[0], color_ptr[1], color_ptr[2]));
                }
            }
            point_ptr += 3;
            if (color_ptr)
                color_ptr += 3;
        }

        cv::Mat validPointsMat(validPoints.size(), 1, CV_32FC3, validPoints.data());
        cv::Mat validColorsMat;
        if (!validColors.empty())
        {
            validColorsMat = cv::Mat(validColors.size(), 1, CV_8UC3, validColors.data());
        }

        cv::viz::writeCloud(filename, validPointsMat, validColorsMat);
        return true;
    }
    catch (const cv::Exception &e)
    {
        std::cerr << "Error saving point cloud: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char **argv)
{

    // load config file
    std::string command_line_keys = "{config | configs/single_test.yaml | path to the config file}"
                                    "{calib | configs/camera_params.yaml | path to the camera parameters file}"
                                    "{test_image | images/test-1.jpg | path to the test image}"
                                    "{output_disp | output/test_disp.exr | path to save the disparity map}"
                                    "{output_pcd | output/test_3d.ply | path to save the point cloud}";
    cv::CommandLineParser parser(argc, argv, command_line_keys);

    std::string conf_path = parser.get<std::string>("config");
    std::string calib_path = parser.get<std::string>("calib");
    std::string img_path = parser.get<std::string>("test_image");
    std::string save_disp_path = parser.get<std::string>("output_disp");
    std::string save_pcd_path = parser.get<std::string>("output_pcd");

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

    // create conversion maps
    std::vector<cv::Mat> left_maps(2);
    std::vector<cv::Mat> right_maps(2);
    cv::Mat map_x, map_y;
    computeStereoRectifyMaps(K1, K2, D1, D2, R1, R2, P1, P2, img_size, left_maps, right_maps, CV_32F);
    computeEquirectangleMaps(P1, img_size, map_x, map_y);

    std::unique_ptr<StereoVisionProcessor> sv_processor = StereoVisionProcessor::create(
        gray_scale, algorithm, blur_kernel, min_disp, max_disp, block_size, p1, p2, max_diff, pre_fc, unique_ratio,
        speckle_size, speckle_range, mode, fx, fy, cx, cy, base_line, map_x, map_y);
    sv_processor->setPostFilter(wsl_lambda, wsl_sigma);

    // Test
    int w = img_size.width;
    int h = img_size.height;
    cv::Mat img, left, right, converted, color;
    cv::Mat disp(h, w, CV_16S);
    cv::Mat disp_norm(h, w, CV_8UC1);
    img = cv::imread(img_path);

    left = img(cv::Rect(0, 0, img.cols / 2, img.rows)).clone();
    right = img(cv::Rect(img.cols / 2, 0, img.cols / 2, img.rows)).clone();
    cv::remap(left, left, left_maps[0], left_maps[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::remap(right, right, right_maps[0], right_maps[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    color = left.clone();
    cv::hconcat(left, right, converted);

    cv::Mat pcd(1, w * h, CV_32FC3);
    cv::Mat colors(1, w * h, CV_8UC3);
    std::vector<bool> valid;

    sv_processor->computePointCloud(left, right, disp, pcd, colors, valid);

    cv::namedWindow("original", cv::WINDOW_NORMAL);
    cv::resizeWindow("original", WINDOW_W, WINDOW_H);
    cv::imshow("original", img);

    cv::namedWindow("converted", cv::WINDOW_NORMAL);
    cv::resizeWindow("converted", WINDOW_W, WINDOW_H);
    cv::imshow("converted", converted);

    cv::normalize(disp, disp_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::namedWindow("disparity", cv::WINDOW_NORMAL);
    cv::resizeWindow("disparity", WINDOW_W, WINDOW_H);
    cv::imshow("disparity", disp_norm);

    cv::waitKey(0);
    cv::destroyAllWindows();

    cv::Mat disp_float;
    disp.convertTo(disp_float, CV_32F);
    cv::imwrite(save_disp_path, disp_float);
    std::cout << "Disparity saved >> " << save_disp_path << std::endl;

    // Save point cloud
    if (!savePointCloudPLY(pcd, colors, valid, save_pcd_path))
    {
        std::cerr << "Failed to save point cloud." << std::endl;
        return -1;
    }
    std::cout << "Point cloud saved >> " << save_pcd_path << std::endl;

    return 0;
}
