#include <cmath>
#include <utils/undistort_utils.h>

namespace undistort_utils
{

void computeStereoRectifyMaps(
  cv::Mat K1, cv::Mat K2, cv::Mat D1, cv::Mat D2,
  cv::Mat R1, cv::Mat R2, cv::Mat P1, cv::Mat P2,
  cv::Size img_size, CameraType camera,
  std::vector<cv::Mat> &left_maps, std::vector<cv::Mat> &right_maps,
  int dtype, cv::Mat xi1, cv::Mat xi2, int omni_type)
{
  switch (camera)
  {
  case CameraType::PINHOLE:
    cv::initUndistortRectifyMap(K1, D1, R1, P1, img_size, dtype,
                                left_maps[0], left_maps[1]);
    cv::initUndistortRectifyMap(K2, D2, R2, P2, img_size, dtype,
                                right_maps[0], right_maps[1]);
    break;
  case CameraType::FISHEYE:
    computeFisheyeMap(K1, D1, R1, P1, img_size, dtype,
                    left_maps[0], left_maps[1]);
    computeFisheyeMap(K2, D2, R2, P2, img_size, dtype,
                    right_maps[0], right_maps[1]);
    break;
  case CameraType::OMNIDIR:
    cv::omnidir::initUndistortRectifyMap(K1, D1, xi1, R1, P1, img_size, dtype,
                                            left_maps[0], left_maps[1], omni_type);
    cv::omnidir::initUndistortRectifyMap(K2, D2, xi2, R2, P2, img_size, dtype,
                                            right_maps[0], right_maps[1], omni_type);
    break;
  default:
    throw std::runtime_error("computeStereoRectifyMap: \"Error invalid camera type.\"");
  }
}

// fisheye image -> (rectified image) -> equirectangular image
void computeFisheyeMap(
  cv::Mat K, cv::Mat D, cv::Mat R, cv::Mat P,
  cv::Size img_size, int dtype, 
  cv::Mat &map_x, cv::Mat &map_y)
{
  cv::Mat rect_map_x(img_size, CV_32FC1);
  cv::Mat rect_map_y(img_size, CV_32FC1);
  cv::Mat rect_to_eqrec_map_x(img_size, CV_32FC1);
  cv::Mat rect_to_eqrec_map_y(img_size, CV_32FC1);
  cv::fisheye::initUndistortRectifyMap(K, D, R, P, img_size, dtype,
                                      rect_map_x, rect_map_y);
  computeEquirectangleMaps(P, img_size,
      rect_to_eqrec_map_x, rect_to_eqrec_map_y);

  cv::remap(rect_map_x, map_x, rect_to_eqrec_map_x, 
          rect_to_eqrec_map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
  cv::remap(rect_map_y, map_y, rect_to_eqrec_map_x, 
          rect_to_eqrec_map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
}

// rectified image -> equirectangular image
void computeEquirectangleMaps(
  cv::Mat K, cv::Size img_size,
  cv::Mat &eqrec_map_x,
  cv::Mat &eqrec_map_y)
{
  eqrec_map_x = cv::Mat(img_size, CV_32FC1);
  eqrec_map_y = cv::Mat(img_size, CV_32FC1);
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);
  double rad_x = std::atan(static_cast<double>(img_size.width)/fx);
  double rad_y = std::atan(static_cast<double>(img_size.height)/fy);
  for (int y = 0; y < img_size.height; y++)
  {
    for (int x = 0; x < img_size.width; x++)
    {
      double lamb = (1.0 - y / (img_size.height / 2.0)) * rad_y;
      double phi = (x / (img_size.width / 2.0) - 1.0) * rad_x;
      double rec_x = cx + fx * tan(phi) / cos(lamb);
      double rec_y = cy - fy * tan(lamb);
      eqrec_map_x.at<float>(y, x) = static_cast<float>(rec_x);
      eqrec_map_y.at<float>(y, x) = static_cast<float>(rec_y);
    }
  }
}

} // namespace undistort_utils
