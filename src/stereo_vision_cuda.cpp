#include <chrono>
#include <stereo_vision_cuda.h>

using namespace std::chrono_literals;

namespace cuda
{

StereoVisionProcessor::StereoVisionProcessor(std::string config_path, std::vector<cv::Mat> left_maps, std::vector<cv::Mat> right_maps)
  : stereo_matcher_(config_path),
    pcd_generator_(config_path),
    left_maps_(left_maps),
    right_maps_(right_maps),
    is_video_cap_open_(false),
    stop_img_update_thread_(true),
    stop_disp_thread_(true),
    stop_pcd_thread_(true),
    stop_viewer_thread_(true),
    stop_pcd_viewer_thread_(true)
{
  config::StereoVisionParams sv_params(config_path);
  setParams(sv_params);

  std::promise<void> img_update_prom, disp_prom, pcd_prom, viewer_prom, pcd_viewer_prom;
  fut_img_update_thread_  = img_update_prom.get_future();
  fut_disp_thread_        = disp_prom.get_future();
  fut_pcd_thread_         = pcd_prom.get_future();
  fut_viewer_thread_      = viewer_prom.get_future();
  fut_pcd_viewer_thread_  = pcd_viewer_prom.get_future();
  img_update_prom.set_value();
  disp_prom.set_value();
  pcd_prom.set_value();
  viewer_prom.set_value();
  pcd_viewer_prom.set_value();
}

void StereoVisionProcessor::setParams(config::StereoVisionParams sv_params)
{
  img_size_             = sv_params.img_size;
  win_size_             = sv_params.win_size;
  img_update_sleep_     = sv_params.img_update_sleep;
  viewer_update_sleep_  = sv_params.viewer_update_sleep;
  source_viewer_        = sv_params.source_viewer;
  rectified_viewer_     = sv_params.rectified_viewer;
  disparity_viewer_     = sv_params.disparity_viewer;
  point_cloud_viewer_   = sv_params.point_cloud_viewer;
  if (point_cloud_viewer_)
  {
    points_size_       = sv_params.points_size;
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
    is_video_cap_open_.store(false);
    return;
  }
  is_video_cap_open_.store(true);

  cap.set(cv::CAP_PROP_BUFFERSIZE, 5);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, img_size_.width * 2);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, img_size_.height);

  // get frame from cap then convert with map
  while (!stop_img_update_thread_.load())
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
  is_video_cap_open_.store(false);
}

void StereoVisionProcessor::computeDisparity()
{
  cv::Mat left, right, disp, left_cp;
  while (!stop_disp_thread_.load())
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
}

void StereoVisionProcessor::computePointCloud()
{
  cv::Mat color, disp;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd(new pcl::PointCloud<pcl::PointXYZRGB>);

  while (!stop_pcd_thread_.load())
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
}

void StereoVisionProcessor::updateViewer()
{
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

  while (!stop_viewer_thread_.load())
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
    {
      cv::imshow("source", src);
    }
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
}

void StereoVisionProcessor::updatePcdViewer()
{
  pcl::visualization::PCLVisualizer viewer("3D viewer");
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd(new pcl::PointCloud<pcl::PointXYZRGB>);

  viewer.setBackgroundColor(0, 0, 0);
  viewer.addCoordinateSystem(coordinate_system_);
  viewer.initCameraParameters();

  while (!stop_pcd_viewer_thread_.load())
  {
    {
      std::shared_lock<std::shared_mutex> lock(mtx_pcd_);
      if (pcd_.empty())
      {
        continue;
      }
      pcl::copyPointCloud(pcd_, *pcd);
    }
    if (!viewer.updatePointCloud(pcd, "pcd"))
    {
      viewer.addPointCloud(pcd, "pcd");
    }
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, points_size_, "pcd");
    viewer.spinOnce(viewer_update_sleep_);
  }
}

bool StereoVisionProcessor::run(int video_cap)
{
  if (!stopThreads())
  {
    return false;
  }

  img_    = cv::Mat::zeros(cv::Size(img_size_.width*2, img_size_.height), CV_8UC3);
  left_   = cv::Mat::zeros(img_size_, CV_8UC3);
  right_  = cv::Mat::zeros(img_size_, CV_8UC3);
  disp_   = cv::Mat::zeros(img_size_, CV_16S);
  rgb_    = cv::Mat::zeros(img_size_, CV_8UC3);

  // launch image update thread first
  std::chrono::milliseconds timeout(5000);
  auto start = std::chrono::high_resolution_clock::now();

  is_video_cap_open_ = false;
  stop_img_update_thread_.store(false);
  fut_img_update_thread_ = std::async(std::launch::async, &StereoVisionProcessor::updateImage, this, video_cap);
  while (!is_video_cap_open_.load())
  {
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
  stop_disp_thread_.store(false);
  fut_disp_thread_ = std::async(std::launch::async, &StereoVisionProcessor::computeDisparity, this);
  std::cout << "Disparity compute thread is running..." << std::endl;

  // image viewer thread
  stop_viewer_thread_.store(false);
  fut_viewer_thread_ = std::async(std::launch::async, &StereoVisionProcessor::updateViewer, this);
  std::cout << "Viewer update thread is running..." << std::endl;

  // pcd generator & pcd viewer thread
  if (point_cloud_viewer_)
  {
    stop_pcd_thread_.store(false);
    fut_pcd_thread_ = std::async(std::launch::async, &StereoVisionProcessor::computePointCloud, this);
    stop_pcd_viewer_thread_.store(false);
    fut_pcd_viewer_thread_ = std::async(std::launch::async, &StereoVisionProcessor::updatePcdViewer, this);
    std::cout << "PointCloud threads are running..." << std::endl;
  }

  return true;
}

bool StereoVisionProcessor::stopThreads()
{
  #define CHECK_THREAD_AND_STOP(thread_name) \
    if (!stop_##thread_name##_thread_.load() && \
        fut_##thread_name##_thread_.wait_for(0ms) == std::future_status::timeout) \
      stop_##thread_name##_thread_.store(true)

  CHECK_THREAD_AND_STOP(img_update);
  CHECK_THREAD_AND_STOP(disp);
  CHECK_THREAD_AND_STOP(pcd);
  CHECK_THREAD_AND_STOP(viewer);
  CHECK_THREAD_AND_STOP(pcd_viewer);

  std::chrono::milliseconds timeout(5000);
  auto start = std::chrono::high_resolution_clock::now();
  while (fut_img_update_thread_.wait_for(0ms) != std::future_status::ready ||
         fut_disp_thread_.wait_for(0ms)       != std::future_status::ready ||
         fut_pcd_thread_.wait_for(0ms)        != std::future_status::ready ||
         fut_viewer_thread_.wait_for(0ms)     != std::future_status::ready ||
         fut_pcd_viewer_thread_.wait_for(0ms) != std::future_status::ready)
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

} // namespace cdua
