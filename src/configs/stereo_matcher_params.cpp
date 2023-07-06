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

    if(conf_map.count("gray_scale")>0) gray_scale = config_utils::getBool(conf_map["gray_scale"]);
    if(conf_map.count("algorithm")>0) algorithm = config_utils::getString(conf_map["algorithm"]);
    if(conf_map.count("blur_kernel")>0) blur_kernel = config_utils::getCvSize<int>(conf_map["blur_kernel"]);
    if(conf_map.count("block_size")>0) block_size = config_utils::getInt(conf_map["block_size"]);
    if(conf_map.count("min_disp")>0) min_disp = config_utils::getInt(conf_map["min_disp"]);
    if(conf_map.count("max_disp")>0) max_disp = config_utils::getInt(conf_map["max_disp"]);
    if(conf_map.count("p1")>0) p1 = config_utils::getInt(conf_map["p1"]);
    if(conf_map.count("p2")>0) p2 = config_utils::getInt(conf_map["p2"]);
    if(conf_map.count("max_diff")>0) max_diff = config_utils::getInt(conf_map["max_diff"]);
    if(conf_map.count("pre_fc")>0) pre_fc = config_utils::getInt(conf_map["pre_fc"]);
    if(conf_map.count("speckle_size")>0) speckle_size = config_utils::getInt(conf_map["speckle_size"]);
    if(conf_map.count("speckle_range")>0) speckle_range = config_utils::getInt(conf_map["speckle_range"]);
    if(conf_map.count("unique_ratio")>0) unique_ratio = config_utils::getInt(conf_map["unique_ratio"]);
    if(conf_map.count("mode")>0) mode = config_utils::getInt(conf_map["mode"]);
}

}
