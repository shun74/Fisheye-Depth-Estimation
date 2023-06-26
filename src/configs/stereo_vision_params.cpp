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

    img_size = config_utils::getCvSize(conf_map["image_size"]);
    win_size = config_utils::getCvSize(conf_map["window_size"]);
    img_update_sleep = config_utils::getInt(conf_map["image_update_sleep"]);
    viewer_update_sleep = config_utils::getInt(conf_map["viewer_update_sleep"]);
    source_viewer = config_utils::getBool(conf_map["source_viewer"]);
    rectified_viewer = config_utils::getBool(conf_map["rectified_viewer"]);
    disparity_viewer = config_utils::getBool(conf_map["disparity_viewer"]);
    point_cloud_viewer = config_utils::getBool(conf_map["point_cloud_viewer"]);
    if (point_cloud_viewer) {
        points_size = config_utils::getInt(conf_map["points_size"]);
        coordinate_system = config_utils::getDouble(conf_map["coordinate_system"]);

    }
}

}