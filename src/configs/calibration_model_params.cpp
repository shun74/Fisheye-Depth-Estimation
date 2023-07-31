#include <utils/config_utils.h>
#include <configs/calibration_model_params.h>

namespace config
{

void CalibrationModelParams::loadParams(std::string path)
{
    std::map<std::string, std::string> conf_map;
    bool load = config_utils::parseConfigFile(path, conf_map);

    if (!load)
    {
        std::cout << "Fail to parse " << path << "." << std::endl;
        return;
    }

    if(conf_map.count("camera_type")>0) cam_type = config_utils::getCameraType(conf_map["camera_type"]);
    if(conf_map.count("image_size")>0) img_size = config_utils::getCvSize<int>(conf_map["image_size"]);
    if(conf_map.count("board_size")>0) board_size = config_utils::getCvSize<int>(conf_map["board_size"]);
    if(conf_map.count("square_size")>0) square_size = config_utils::getDouble(conf_map["square_size"]);
    if(conf_map.count("fov")>0) fov = config_utils::getDouble(conf_map["fov"]);
}

} // namespace config