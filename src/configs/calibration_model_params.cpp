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

        cam_type = config_utils::getCameraType(conf_map["camera_type"]);
        img_size = config_utils::getCvSize<int>(conf_map["image_size"]);
        board_size = config_utils::getCvSize<int>(conf_map["board_size"]);
        square_size = config_utils::getDouble(conf_map["square_size"]);
        fov = config_utils::getDouble(conf_map["fov"]);
    }

}