#include <chrono>
#include <stereo_vision_cuda.h>

namespace cuda
{

StereoVisionProcessor::StereoVisionProcessor(std::string path, std::vector<cv::Mat> left_maps, std::vector<cv::Mat> right_maps)
    : stereo_matcher_(path), pcd_generator_(path)
{
    video_cap_ = false;
    stop_img_update_ = true;
    up_img_update_ = false;
    stop_disp_ = true;
    up_disp_ = false;
    stop_pcd_ = true;
    up_pcd_ = false;
    stop_viewer_ = true;
    up_viewer_ = false;
    stop_pcd_viewer_ = true;
    up_pcd_viewer_ = false;

    left_maps_ = left_maps;
    right_maps_ = right_maps;

    config::StereoVisionParams sv_params(path);
    setParams(sv_params);
}

void StereoVisionProcessor::setParams(config::StereoVisionParams sv_params)
{
    img_size_ = sv_params.img_size;
    win_size_ = sv_params.win_size;
    img_update_sleep_ = sv_params.img_update_sleep;
    viewer_update_sleep_ = sv_params.viewer_update_sleep;
    source_viewer_ = sv_params.source_viewer;
    rectified_viewer_ = sv_params.rectified_viewer;
    disparity_viewer_ = sv_params.disparity_viewer;
    point_cloud_viewer_ = sv_params.point_cloud_viewer;
    if (point_cloud_viewer_)
    {
        points_size_ = sv_params.points_size;
        coordinate_system_ = sv_params.coordinate_system;
    }
}

void StereoVisionProcessor::setConvertMaps(cv::Mat map_x, cv::Mat map_y)
{
    pcd_generator_.setConvertMaps(map_x, map_y);
}

void StereoVisionProcessor::updateImage(int video_cap)
{
    // get video cap
    cv::VideoCapture cap(video_cap);
    if (!cap.isOpened())
    {
        video_cap_ = false;
        return;
    }
    video_cap_ = true;
    up_img_update_ = true;

    cap.set(cv::CAP_PROP_BUFFERSIZE, 5);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, img_size_.width * 2);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, img_size_.height);

    // get frame from cap then convert with map
    while (!stop_img_update_)
    {
        {
            std::lock_guard<std::shared_mutex> lock(mtx_img_);
            cap >> img_;
            img_(cv::Rect(0, 0, img_.cols / 2, img_.rows)).copyTo(left_);
            img_(cv::Rect(img_.cols / 2, 0, img_.cols / 2, img_.rows)).copyTo(right_);
            cv::remap(left_, left_, left_maps_[0], left_maps_[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
            cv::remap(right_, right_, right_maps_[0], right_maps_[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(img_update_sleep_));
    }
    cap.release();
    video_cap_ = false;
    up_img_update_ = false;
    stop_img_update_ = false;
}

void StereoVisionProcessor::computeDisparity()
{
    up_disp_ = true;
    cv::Mat left, right, disp, left_cp;
    while (!stop_disp_)
    {
        {
            std::shared_lock<std::shared_mutex> lock(mtx_img_);
            left_.copyTo(left);
            right_.copyTo(right);
        }
        // copy left for pcd color
        left.copyTo(left_cp);

        stereo_matcher_.computeDisparity(left, right, disp);

        {
            std::lock_guard<std::shared_mutex> lock(mtx_disp_);
            disp.copyTo(disp_);
            left_cp.copyTo(rgb_);
        }
    }
    up_disp_ = false;
    stop_disp_ = false;
}

void StereoVisionProcessor::computePointCloud()
{
    up_pcd_ = true;
    cv::Mat color, disp;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd(new pcl::PointCloud<pcl::PointXYZRGB>);

    while (!stop_pcd_)
    {
        {
            std::shared_lock<std::shared_mutex> lock(mtx_disp_);
            disp_.copyTo(disp);
            rgb_.copyTo(color);
        }

        pcd_generator_.computePointCloud(color, disp, pcd);

        if (pcd->empty())
            continue;
        {
            std::lock_guard<std::shared_mutex> lock(mtx_pcd_);
            pcl::copyPointCloud(*pcd, pcd_);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(img_update_sleep_));
    }
    up_pcd_ = false;
    stop_pcd_ = false;
}

void StereoVisionProcessor::updateViewer()
{
    up_viewer_ = true;
    cv::Mat src, left, right, rect, disp, disp_norm;

    if (source_viewer_)
    {
        cv::namedWindow("source", cv::WINDOW_NORMAL);
        cv::resizeWindow("source", win_size_.width, win_size_.height);
    }
    if (rectified_viewer_)
    {
        cv::namedWindow("rectified", cv::WINDOW_NORMAL);
        cv::resizeWindow("rectified", win_size_.width, win_size_.height);
    }
    if (disparity_viewer_)
    {
        cv::namedWindow("disparity", cv::WINDOW_NORMAL);
        cv::resizeWindow("disparity", win_size_.width / 2, win_size_.height);
    }

    while (!stop_viewer_)
    {
        {
            std::shared_lock<std::shared_mutex> lock(mtx_img_);
            img_.copyTo(src);
            left_.copyTo(left);
            right_.copyTo(right);
        }
        {
            std::shared_lock<std::shared_mutex> lock(mtx_disp_);
            disp_.copyTo(disp);
        }

        if (source_viewer_)
            cv::imshow("source", src);
        if (rectified_viewer_)
        {
            cv::hconcat(left, right, rect);
            cv::imshow("rectified", rect);
        }
        if (disparity_viewer_)
        {
            cv::normalize(disp, disp_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::imshow("disparity", disp_norm);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(viewer_update_sleep_));
    }
    cv::destroyAllWindows();
    up_viewer_ = false;
    stop_viewer_ = false;
}

void StereoVisionProcessor::updatePcdViewer()
{
    up_pcd_viewer_ = true;
    pcl::visualization::PCLVisualizer viewer("3D viewer");
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd(new pcl::PointCloud<pcl::PointXYZRGB>);

    viewer.setBackgroundColor(0, 0, 0);
    viewer.addCoordinateSystem(coordinate_system_);
    viewer.initCameraParameters();

    while (!stop_pcd_viewer_)
    {
        {
            std::shared_lock<std::shared_mutex> lock(mtx_pcd_);
            if (pcd_.empty())
                continue;
            pcl::copyPointCloud(pcd_, *pcd);
        }
        if (!viewer.updatePointCloud(pcd, "pcd"))
        {
            viewer.addPointCloud(pcd, "pcd");
        }
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, points_size_, "pcd");
        viewer.spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(viewer_update_sleep_));
    }
    up_pcd_viewer_ = false;
    stop_pcd_viewer_ = false;
}

bool StereoVisionProcessor::run(int video_cap)
{
    if (!stopThreads())
        return false;

    img_ = cv::Mat::zeros(cv::Size(img_size_.width*2, img_size_.height), CV_8UC3);
    left_ = cv::Mat::zeros(img_size_, CV_8UC3);
    right_ = cv::Mat::zeros(img_size_, CV_8UC3);
    disp_ = cv::Mat::zeros(img_size_, CV_16S);
    rgb_ = cv::Mat::zeros(img_size_, CV_8UC3);

    // launch image update thread first
    std::chrono::milliseconds timeout(5000);
    auto start = std::chrono::high_resolution_clock::now();
    video_cap_ = false;
    th_img_update_ = std::thread(&StereoVisionProcessor::updateImage, this, video_cap);
    while (!video_cap_)
    {
        if (th_img_update_.joinable())
            th_img_update_.detach();
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start);
        if (elapsed >= timeout)
        {
            std::cout << "Opening video capture timeout: (" << video_cap << ")" << std::endl;
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "Image Update thread is running..." << std::endl;

    // disparity thread
    th_disp_ = std::thread(&StereoVisionProcessor::computeDisparity, this);
    th_disp_.detach();
    std::cout << "Disparity compute thread is running..." << std::endl;

    // image viewer thread
    th_viewer_ = std::thread(&StereoVisionProcessor::updateViewer, this);
    th_viewer_.detach();
    std::cout << "Viewer update thread is running..." << std::endl;

    // pcd generator & pcd viewer thread
    if (point_cloud_viewer_)
    {
        th_pcd_ = std::thread(&StereoVisionProcessor::computePointCloud, this);
        th_pcd_.detach();
        th_pcd_viewer_ = std::thread(&StereoVisionProcessor::updatePcdViewer, this);
        th_pcd_viewer_.detach();
        std::cout << "PointCloud threads are running..." << std::endl;
    }

    return true;
}

bool StereoVisionProcessor::stopThreads()
{
    if (!stop_img_update_ && !th_img_update_.joinable() && up_img_update_)
        stop_img_update_ = true;
    else
        stop_img_update_ = false;
    if (!stop_disp_ && !th_disp_.joinable() && up_disp_)
        stop_disp_ = true;
    else
        stop_disp_ = false;
    if (!stop_pcd_ && !th_pcd_.joinable() && up_pcd_)
        stop_pcd_ = true;
    else
        stop_pcd_ = false;
    if (!stop_viewer_ && !th_viewer_.joinable() && up_viewer_)
        stop_viewer_ = true;
    else
        stop_viewer_ = false;
    if (!stop_pcd_viewer_ && !th_pcd_viewer_.joinable() && up_pcd_viewer_)
        stop_pcd_viewer_ = true;
    else
        stop_pcd_viewer_ = false;

    std::chrono::milliseconds timeout(5000);
    auto start = std::chrono::high_resolution_clock::now();
    while (stop_img_update_ || stop_disp_ || stop_pcd_ || stop_viewer_ || stop_pcd_viewer_)
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start);
        if (elapsed >= timeout)
        {
            std::cout << "_stopThreads timeout." << std::endl;
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return true;
}

}