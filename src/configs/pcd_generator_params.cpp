#include <utils/config_utils.h>
#include <configs/pcd_generator_params.h>

namespace config
{

void PointCloudGeneratorParams::loadParams(std::string path)
{
  std::map<std::string, std::string> conf_map;

  bool load = config_utils::parseConfigFile(path, conf_map);

  if (!load) {
    std::cout << "Fail to parse " << path << "." << std::endl;
    return;
  }

  if (conf_map.count("is_fisheye")>0) is_fisheye = config_utils::getBool(conf_map["is_fisheye"]);
  if (conf_map.count("x_clip")>0) {
    cv::Size_<double> x_clip = config_utils::getCvSize<double>(conf_map["x_clip"]);
    x_min = x_clip.width;
    x_max = x_clip.height;
  }
  if (conf_map.count("y_clip")>0) {
    cv::Size_<double> y_clip = config_utils::getCvSize<double>(conf_map["y_clip"]);
    y_min = y_clip.width;
    y_max = y_clip.height;
  }
  if (conf_map.count("z_clip")>0) {
    cv::Size_<double> z_clip = config_utils::getCvSize<double>(conf_map["z_clip"]);
    z_min = z_clip.width;
    z_max = z_clip.height;
  }
  if (conf_map.count("use_down_sample")>0) use_down_sample = config_utils::getBool(conf_map["use_down_sample"]);
  if (use_down_sample) {
    if (conf_map.count("leaf_x")>0) leaf_x = config_utils::getDouble(conf_map["leaf_x"]);
    if (conf_map.count("leaf_y")>0) leaf_y = config_utils::getDouble(conf_map["leaf_x"]);
    if (conf_map.count("leaf_z")>0) leaf_z = config_utils::getDouble(conf_map["leaf_x"]);

  }

  std::string calib_path;
  if (conf_map.count("calibration_file")>0) {
    calib_path = config_utils::getString(conf_map["calibration_file"]);
    cv::FileStorage fs(calib_path, cv::FileStorage::READ);
    cv::FileNode cam_params = fs["camera_params"];
    cv::Mat K, T;
    cam_params["P1"] >> K; cam_params["T"] >> T;
    fx = K.at<double>(0,0);
    fy = K.at<double>(1,1);
    cx = K.at<double>(0,2);
    cy = K.at<double>(1,2);
    base = -T.at<double>(0);
  }
}

} // namespace config
