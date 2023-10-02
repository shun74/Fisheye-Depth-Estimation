#include <csignal>
#include <configs/camera_type.h>
#include <utils/config_utils.h>
#include <utils/undistort_utils.h>
#include <stereo_vision_cuda.h>

bool exit_flag = false;

void handle_signal(int signal) {
  std::cout << "\n" << "Caught signal " << signal << std::endl;
  exit_flag = true;
}

int main(int argc, char** argv) {
  // monitor the signales to stop
  std::signal(SIGINT, handle_signal);

  // load config file
  std::string conf_path = "configs/realtime_stereo_cuda.conf";
  std::string calib_path = "configs/camera_params.yaml";
  std::map<std::string, std::string> args_map;
  args_map = config_utils::parseArguments(argc, argv);
  if (args_map.find("config")!=args_map.end()) conf_path = args_map["config"];
  if (args_map.find("calib")!=args_map.end()) calib_path = args_map["calib"];

  // load camera parameters file
  cv::FileStorage fs(calib_path, cv::FileStorage::READ);
  cv::FileNode cp = fs["camera_params"];
  CameraType cam_type = config_utils::getCameraType(fs["camera_type"]);
  double fov = fs["fov"];

  cv::Size img_size;
  fs["img_size"] >> img_size;
  cv::Mat K1, K2, D1, D2, R1, R2, P1, P2, T;
  cp["K1"] >> K1; cp["K2"] >> K2; cp["D1"] >> D1; cp["D2"] >> D2;
  cp["R1"] >> R1; cp["R2"] >> R2; cp["P1"] >> P1; cp["P2"] >> P2;

  // create conversion maps
  std::vector<cv::Mat> left_maps(2);
  std::vector<cv::Mat> right_maps(2);

  if (cam_type!=CameraType::OMNIDIR) {
    undistort_utils::computeStereoRectifyMaps(
      K1, K2, D1, D2,
      R1, R2, P1, P2,
      img_size, cam_type, left_maps, right_maps
    );
  }
  else {
    cv::Mat xi1, xi2;
    cp["xi1"] >> xi1; cp["xi2"] >> xi2;
    undistort_utils::computeStereoRectifyMaps(
      K1, K2, D1, D2,
      R1, R2, P1, P2,
      img_size, cam_type, left_maps, right_maps,
      CV_32F, xi1, xi2
    );
  }

  cuda::StereoVisionProcessor sv_processor = cuda::StereoVisionProcessor(conf_path, left_maps, right_maps);

  if (cam_type==CameraType::FISHEYE) {
    cv::Mat map_x, map_y;
    undistort_utils::computeEquirectangleMaps(P1, img_size, map_x, map_y);
    sv_processor.setConvertMaps(map_x, map_y);
  }

  // run all threads
  int video_cap = 0;
  if(!sv_processor.run(video_cap)) return -1;

  // monitor key inputs and signels to stop
  while (!exit_flag) {
    if (cv::waitKey(1)>=0) exit_flag = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  
  return 0;
}
