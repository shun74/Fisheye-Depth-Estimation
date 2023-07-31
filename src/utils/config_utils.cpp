#include <utils/config_utils.h>

namespace config_utils
{

bool getBool(std::string s)
{
    // "true" -> true
    return s == "true";
}

int getInt(std::string s)
{
    return std::stoi(s);
}

double getDouble(std::string s)
{
    return std::stod(s);
}

std::string getString(std::string s)
{
    // "\"abc\"" -> "abc"
    s.erase(std::remove(s.begin(), s.end(), '\"'), s.end());
    return s;
}

CameraType getCameraType(std::string s)
{
    s.erase(std::remove(s.begin(), s.end(), '\"'), s.end());
    CameraType type;
    if (s == "pinhole")
        type = CameraType::PINHOLE;
    else if (s == "fisheye")
        type = CameraType::FISHEYE;
    else if (s == "omnidir")
        type = CameraType::OMNIDIR;
    else
        throw std::runtime_error("CarlibrationModel: Invalid CameraType detected.");
    return type;
}

bool parseConfigFile(std::string path, std::map<std::string, std::string> &conf_map)
{
    std::ifstream ifs(path);

    if (ifs.fail())
    {
        std::cerr << "Failed to open file." << std::endl;
        return false;
    }

    std::string line;
    int col = 0;

    while (getline(ifs, line))
    {
        col++;

        std::size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos)
            line.erase(comment_pos);
        if (line.length() == 0)
            continue;

        std::vector<std::string> block;
        std::stringstream ss(line);
        std::string s;

        while (getline(ss, s, '='))
        {
            s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
            block.push_back(s);
        }
        if (block.size() != 2)
        {
            std::cout << "Invalid config in line: " << col << "." << std::endl;
            return false;
        }
        conf_map.insert(std::make_pair(block[0], block[1]));
    }

    return true;
}

std::map<std::string, std::string> parseArguments(int argc, char **argv)
{
    std::map<std::string, std::string> arg_map;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        auto pos = arg.find('=');
        // pattern: --image=./path
        if (pos != std::string::npos)
        {
            std::string key = arg.substr(0, pos);
            std::string value = arg.substr(pos + 1);

            if (key[0] == '-')
            {
                key = key.substr(key[1] == '-' ? 2 : 1);
            }
            arg_map[key] = value;
        }
        // pattern: --image ./path
        else if (i + 1 < argc)
        {
            std::string key = arg;
            std::string value = argv[++i];

            if (key[0] == '-')
            {
                key = key.substr(key[1] == '-' ? 2 : 1);
            }
            arg_map[key] = value;
        }
        else
        {
            throw std::invalid_argument("未知の引数: " + arg);
        }
    }
    return arg_map;
}

} // namespace config_utils
