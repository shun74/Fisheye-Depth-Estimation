#include <iostream>
#include <pcl/io/pcd_io.h>
#include <configs/camera_type.h>
#include <utils/config_utils.h>
#include <utils/undistort_utils.h>
#include <engines/stereo_matcher.h>
#include <engines/pcd_generator.h>


int main(int argc, char** argv) {

    std::string img_path = "images/test-1.jpg";
    if (argc>1) img_path = argv[1];

    std::string conf_path = "configs/test.conf";
    if (argc>2) conf_path = argv[2];

    std::string calib_path = "configs/camera_params.yaml";
    if (argc>3) calib_path = argv[3];

    // prepare for test
    cv::FileStorage fs(calib_path, cv::FileStorage::READ);
    cv::FileNode cp = fs["camera_params"];
    CameraType cam_type = config_utils::getCameraType(fs["camera_type"]);
    if (cam_type!=CameraType::FISHEYE) {
        std::cout << "Fisheye test doesn't allow another type." << std::endl;
        return -1;
    }
    double fov = fs["fov"];

    cv::Size img_size;
    fs["img_size"] >> img_size;
    cv::Mat K1, K2, D1, D2, R1, R2, P1, P2, T;
    cp["K1"] >> K1; cp["K2"] >> K2; cp["D1"] >> D1; cp["D2"] >> D2;
    cp["R1"] >> R1; cp["R2"] >> R2; cp["P1"] >> P1; cp["P2"] >> P2;

    std::vector<cv::Mat> rect_left_maps(2);
    std::vector<cv::Mat> rect_right_maps(2);
    std::vector<cv::Mat> eqrec_left_maps(2);
    std::vector<cv::Mat> eqrec_right_maps(2);

    cv::fisheye::initUndistortRectifyMap(K1, D1, R1, P1, img_size, CV_32F,
                                        rect_left_maps[0], rect_left_maps[1]);
    cv::fisheye::initUndistortRectifyMap(K2, D2, R2, P2, img_size, CV_32F,
                                        rect_right_maps[0], rect_right_maps[1]);

    undistort_utils::computeStereoRectifyMaps(
        K1, K2, D1, D2, R1, R2, P1, P2,
        img_size, cam_type, eqrec_left_maps, eqrec_right_maps);
    cv::Mat eqrec_to_rect_map_x, eqrec_to_rect_map_y;
    undistort_utils::computeEquirectangleMaps(P1, img_size, eqrec_to_rect_map_x, eqrec_to_rect_map_y);

    engine::StereoMatcher stereo_matcher(conf_path);

    // Test
    cv::namedWindow("original", cv::WINDOW_NORMAL);
    cv::resizeWindow("original", 720, 480);
    cv::namedWindow("rectified", cv::WINDOW_NORMAL);
    cv::resizeWindow("rectified", 360, 360);
    cv::namedWindow("equirectangular", cv::WINDOW_NORMAL);
    cv::resizeWindow("equirectangular", 360, 360);
    cv::namedWindow("rect_disp", cv::WINDOW_NORMAL);
    cv::resizeWindow("rect_disp", 360, 360);
    cv::namedWindow("eqrec_disp", cv::WINDOW_NORMAL);
    cv::resizeWindow("eqrec_disp", 360, 360);

    cv::Mat img, rect_left, rect_right, eqrec_left, eqrec_right, disp, rect_disp_norm, eqrec_disp_norm;

    img = cv::imread(img_path);

    rect_left = img(cv::Rect(0,0,img.cols/2,img.rows)).clone();
    rect_right = img(cv::Rect(img.cols/2,0,img.cols/2,img.rows)).clone();
    cv::remap(rect_left, rect_left, rect_left_maps[0], rect_left_maps[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::remap(rect_right, rect_right, rect_right_maps[0], rect_right_maps[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::imwrite("output/rect.png", rect_left);
    stereo_matcher.computeDisparity(rect_left, rect_right, disp);
    cv::normalize(disp, rect_disp_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite("output/rect_disp.png", rect_disp_norm);

    eqrec_left = img(cv::Rect(0,0,img.cols/2,img.rows)).clone();
    eqrec_right = img(cv::Rect(img.cols/2,0,img.cols/2,img.rows)).clone();
    cv::remap(eqrec_left, eqrec_left, eqrec_left_maps[0], eqrec_left_maps[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::remap(eqrec_right, eqrec_right, eqrec_right_maps[0], eqrec_right_maps[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::imwrite("output/eqrec.png", eqrec_left);
    stereo_matcher.computeDisparity(eqrec_left, eqrec_right, disp);
    cv::normalize(disp, eqrec_disp_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite("output/eqrec_disp.png", eqrec_disp_norm);

    cv::imshow("original", img);
    cv::imshow("rectified", rect_left);
    cv::imshow("equirectangular", eqrec_left);
    cv::imshow("rect_disp", rect_disp_norm);
    cv::imshow("eqrec_disp", eqrec_disp_norm);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}