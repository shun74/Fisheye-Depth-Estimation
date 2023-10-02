#include <utils/config_utils.h>
#include <configs/stereo_vision_params.h>

namespace config
{

void StereoVisionParams::loadParams(std::string path)
{
  std::map<std::string, std::string> conf_map;

  bool load = config_utils::parseConfigFile(path, conf_map);
  
  if (!load) {
    std::cout << "Fail to parse " << path << "." << std::endl;
    return;
  }

  if(conf_map.count("image_size")>0) img_size = config_utils::getCvSize<int>(conf_map["image_size"]);
  if(conf_map.count("window_size")>0) win_size = config_utils::getCvSize<int>(conf_map["window_size"]);
  if(conf_map.count("image_update_sleep")>0) img_update_sleep = config_utils::getInt(conf_map["image_update_sleep"]);
  if(conf_map.count("viewer_update_sleep")>0) viewer_update_sleep = config_utils::getInt(conf_map["viewer_update_sleep"]);
  if(conf_map.count("source_viewer")>0) source_viewer = config_utils::getBool(conf_map["source_viewer"]);
  if(conf_map.count("rectified_viewer")>0) rectified_viewer = config_utils::getBool(conf_map["rectified_viewer"]);
  if(conf_map.count("disparity_viewer")>0) disparity_viewer = config_utils::getBool(conf_map["disparity_viewer"]);
  if(conf_map.count("point_cloud_viewer")>0) point_cloud_viewer = config_utils::getBool(conf_map["point_cloud_viewer"]);
  if (point_cloud_viewer) {
    if(conf_map.count("points_size")>0) points_size = config_utils::getInt(conf_map["points_size"]);
    if(conf_map.count("coordinate_system")>0) coordinate_system = config_utils::getDouble(conf_map["coordinate_system"]);
  }
}

} // namespace config
