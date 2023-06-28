#include <utils/config_utils.h>
#include <configs/stereo_matcher_params.h>

namespace config
{

void StereoMatcherParams::loadParams(std::string path)
{
    std::map<std::string, std::string> conf_map;

    bool load = config_utils::parseConfigFile(path, conf_map);
    
    if (!load) {
        std::cout << "Fail to parse " << path << "." << std::endl;
        return;
    }

    gray_scale = config_utils::getBool(conf_map["gray_scale"]);
    algorithm = config_utils::getString(conf_map["algorithm"]);
    blur_kernel = config_utils::getCvSize<int>(conf_map["blur_kernel"]);
    block_size = config_utils::getInt(conf_map["block_size"]);
    min_disp = config_utils::getInt(conf_map["min_disp"]);
    max_disp = config_utils::getInt(conf_map["max_disp"]);
    p1 = config_utils::getInt(conf_map["p1"]);
    p2 = config_utils::getInt(conf_map["p2"]);
    max_diff = config_utils::getInt(conf_map["max_diff"]);
    pre_fc = config_utils::getInt(conf_map["pre_fc"]);
    speckle_size = config_utils::getInt(conf_map["speckle_size"]);
    speckle_range = config_utils::getInt(conf_map["speckle_range"]);
    unique_ratio = config_utils::getInt(conf_map["unique_ratio"]);
    mode = config_utils::getInt(conf_map["mode"]);
}

}
