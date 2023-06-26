#include <configs/camera_type.h>
#include <engines/calibration_model.h>
#include <utils/config_utils.h>
#include <utils/undistort_utils.h>

int main (int argc, char** argv) {

    std::string conf_path = "configs/calibration.conf";
    if (argc>1) conf_path = argv[1];

    std::string imgs_dir = "calib_images/";
    if (argc>2) imgs_dir = argv[2];

    std::string save_path = "configs/camera_params.yaml";
    if (argc>3) save_path = argv[3];

    std::string test_path = "images/test-1.jpg";
    if (argc>4) test_path = argv[4];

    engine::CalibrationModel calib_model = engine::CalibrationModel(conf_path);

    std::vector<cv::Mat> left_imgs, right_imgs;
    engine::stereoImageLoader(imgs_dir, left_imgs, right_imgs);

    calib_model.calibrate(left_imgs, right_imgs);
    calib_model.writeCameraParams(save_path);

    cv::FileStorage fs(save_path, cv::FileStorage::READ);

    std::string cam_type_str;
    fs["camera_type"] >> cam_type_str;
    CameraType cam_type;
    cam_type = config_utils::getCameraType(cam_type_str);
    cv::Size img_size;
    fs["img_size"] >> img_size;
    
    cv::FileNode cp = fs["camera_params"];
    cv::Mat K1, K2, D1, D2, R1, R2, P1, P2;
    cp["K1"] >> K1; cp["K2"] >> K2; cp["D1"] >> D1; cp["D2"] >> D2;
    cp["R1"] >> R1; cp["R2"] >> R2; cp["P1"] >> P1; cp["P2"] >> P2;

    std::vector<cv::Mat> left_maps(2);
    std::vector<cv::Mat> right_maps(2);
    if (cam_type!=CameraType::OMNIDIR) {
        undistort_utils::computeStereoRectifyMaps(
            K1, K2, D1, D2, R1, R2, P1, P2,
            img_size, cam_type, left_maps, right_maps);
    }
    else {
        cv::Mat xi1, xi2;
        cp["xi1"] >> xi1; cp["xi2"] >> xi2;
        undistort_utils::computeStereoRectifyMaps(
            K1, K2, D1, D2, R1, R2, P1, P2,
            img_size, cam_type, left_maps, right_maps,
            CV_32F, xi1, xi2);
    }

    cv::Mat test_img = cv::imread(test_path);
    cv::Mat left = test_img(cv::Rect(0,0,test_img.cols/2,test_img.rows)).clone();
    cv::Mat right = test_img(cv::Rect(test_img.cols/2,0,test_img.cols/2,test_img.rows)).clone();

    cv::remap(left, left, left_maps[0], left_maps[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::remap(right, right, right_maps[0], right_maps[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    cv::Mat converted;
    cv::hconcat(left, right, converted);
    cv::line(converted, cv::Point(0, 480), cv::Point(2560, 480), cv::Scalar(0,255,0), 2);

    cv::namedWindow("original", cv::WINDOW_NORMAL);
    cv::resizeWindow("original", 1280, 480);
    cv::imshow("original", test_img);
    cv::namedWindow("calib", cv::WINDOW_NORMAL);
    cv::resizeWindow("calib", 1280, 480);
    cv::imshow("calib", converted);
    cv::waitKey(0);

    return 0;
}