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

    std::string save_disp_path = "output/test_disp.exr"; // OpenEXR recommanded
    if (argc>2) save_disp_path = argv[2];

    std::string save_pcd_path = "output/test_3d.pcd";
    if (argc>3) save_pcd_path = argv[3];

    std::string conf_path = "configs/test.conf";
    if (argc>4) conf_path = argv[4];

    std::string calib_path = "configs/camera_params.yaml";
    if (argc>5) calib_path = argv[5];

    // prepare for test
    cv::FileStorage fs(calib_path, cv::FileStorage::READ);
    cv::FileNode cp = fs["camera_params"];
    CameraType cam_type = config_utils::getCameraType(fs["camera_type"]);
    double fov = fs["fov"];

    cv::Size img_size;
    fs["img_size"] >> img_size;
    cv::Mat K1, K2, D1, D2, R1, R2, P1, P2, T;
    cp["K1"] >> K1; cp["K2"] >> K2; cp["D1"] >> D1; cp["D2"] >> D2;
    cp["R1"] >> R1; cp["R2"] >> R2; cp["P1"] >> P1; cp["P2"] >> P2;

    std::vector<cv::Mat> left_maps(2);
    std::vector<cv::Mat> right_maps(2);

    if (cam_type!=CameraType::OMNIDIR) {
        undistort_utils::computeStereoRectifyMaps(
            K1, K2, D1, D2,
            R1, R2, P1, P2,
            img_size, cam_type, left_maps, right_maps
        );
    }
    else {
        cv::Mat xi1, xi2;
        cp["xi1"] >> xi1; cp["xi2"] >> xi2;
        undistort_utils::computeStereoRectifyMaps(
            K1, K2, D1, D2,
            R1, R2, P1, P2,
            img_size, cam_type, left_maps, right_maps,
            CV_32F, xi1, xi2
        );
    }

    engine::StereoMatcher stereo_matcher(conf_path);
    engine::PointCloudGenerator pcd_generator(conf_path);

    if (cam_type==CameraType::FISHEYE) {
        cv::Mat map_x, map_y;
        undistort_utils::computeEquirectangleMaps(P1, img_size, map_x, map_y);
        pcd_generator.setConvertMaps(map_x, map_y);
    }

    // Test
    cv::namedWindow("original", cv::WINDOW_NORMAL);
    cv::resizeWindow("original", 720, 480);
    cv::namedWindow("converted", cv::WINDOW_NORMAL);
    cv::resizeWindow("converted", 720, 480);
    cv::namedWindow("disparity", cv::WINDOW_NORMAL);
    cv::resizeWindow("disparity", 360, 360);
    cv::Mat img, left, right, converted, disp, disp_norm, color;
    img = cv::imread(img_path);

    left = img(cv::Rect(0,0,img.cols/2,img.rows)).clone();
    right = img(cv::Rect(img.cols/2,0,img.cols/2,img.rows)).clone();
    cv::remap(left, left, left_maps[0], left_maps[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::remap(right, right, right_maps[0], right_maps[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    color = left.clone();
    
    cv::hconcat(left, right, converted);
  
    stereo_matcher.computeDisparity(left, right, disp);
    cv::Mat disp_fp32(disp.size(), CV_32FC1);
    for (int y=0; y<disp.size().height; y++) {
        for (int x=0; x<disp.size().width; x++) {
            disp_fp32.at<float>(y,x) = static_cast<float>(disp.at<short>(y,x))/16.0f;
        }
    }

    cv::normalize(disp, disp_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("original", img);
    cv::imshow("converted", converted);
    cv::imshow("disparity", disp_norm);
    cv::waitKey(0);
    cv::destroyAllWindows();
    cv::imwrite(save_disp_path, disp_fp32);
    std::cout << "Disparity saved >> " << save_disp_path << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcd_generator.computePointCloud(color, disp, pcd);
    pcl::io::savePCDFileASCII(save_pcd_path, *pcd);
    std::cout << "Point Cloud saved >> " << save_pcd_path << std::endl;

    return 0;
}