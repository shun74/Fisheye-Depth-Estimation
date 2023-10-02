#include <utility>
#include <filesystem>
#include <stdexcept>
#include <engines/calibration_model.h>

namespace engine
{

void stereoImageLoader(
  std::string img_dir, std::vector<cv::Mat> &left_imgs, std::vector<cv::Mat> &right_imgs)
{
  std::filesystem::path dir = img_dir;

  for (const auto &entry : std::filesystem::directory_iterator(dir))
  {
    if (entry.is_regular_file())
    {
      cv::Mat image = cv::imread(entry.path(), cv::IMREAD_GRAYSCALE);
      cv::Mat left = image(cv::Rect(0, 0, image.cols / 2, image.rows));
      cv::Mat right = image(cv::Rect(image.cols / 2, 0, image.cols / 2, image.rows));
      left_imgs.push_back(left);
      right_imgs.push_back(right);
    }
  }
}

CalibrationModel::CalibrationModel(std::string path)
{
  config::CalibrationModelParams cm_params(path);
  setParams(cm_params);
  is_calibrated_ = false;
}

CalibrationModel::CalibrationModel(config::CalibrationModelParams cm_params)
{
  setParams(cm_params);
  is_calibrated_ = false;
}

void CalibrationModel::scanCheckerBoard(
  std::vector<cv::Mat> &left_imgs, std::vector<cv::Mat> &right_imgs,
  std::vector<std::vector<cv::Point2f>> &left_points,
  std::vector<std::vector<cv::Point2f>> &right_points)
{
  int grid_num = board_size_.width * board_size_.height;
  for (int i = 0; i < left_imgs.size(); i++)
  {
    std::vector<cv::Point2f> left_corners;
    std::vector<cv::Point2f> right_corners;
    bool found1 = cv::findChessboardCorners(left_imgs[i], board_size_, left_corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
    bool found2 = cv::findChessboardCorners(right_imgs[i], board_size_, right_corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
    if (found1 && found2 && left_corners.size() == grid_num && right_corners.size() == grid_num)
    {
      cv::Size subpix_window(7, 7);
      cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1);
      cv::cornerSubPix(left_imgs[i], left_corners, subpix_window, cv::Size(-1, -1), criteria);
      cv::cornerSubPix(right_imgs[i], right_corners, subpix_window, cv::Size(-1, -1), criteria);
      left_points.push_back(left_corners);
      right_points.push_back(right_corners);
    }
  }
}

void CalibrationModel::getObjPoints(
  int num, std::vector<std::vector<cv::Point3f>> &obj_points)
{
  for (int m = 0; m < num; m++)
  {
    std::vector<cv::Point3f> points;
    for (int j = 0; j < board_size_.height; j++)
    {
      for (int i = 0; i < board_size_.width; i++)
      {
        points.push_back(cv::Point3f(i * square_size_, j * square_size_, 0));
      }
    }
    obj_points.push_back(points);
  }
}

void CalibrationModel::calibratePinhole(
  std::vector<std::vector<cv::Point2f>> left_points,
  std::vector<std::vector<cv::Point2f>> right_points,
  std::vector<std::vector<cv::Point3f>> obj_points)
{
  cv::Mat E, F, Q;
  double rms = cv::stereoCalibrate(obj_points, left_points, right_points,
                                    K1_, D1_, K2_, D2_, img_size_, R_, T_, E, F);
  std::cout << "Calib error: " << rms << std::endl;
  cv::stereoRectify(K1_, D1_, K2_, D2_, img_size_,
                    R_, T_, R1_, R2_, P1_, P2_, Q);
}

void CalibrationModel::calibrateFisheye(
  std::vector<std::vector<cv::Point2f>> left_points,
  std::vector<std::vector<cv::Point2f>> right_points,
  std::vector<std::vector<cv::Point3f>> obj_points)
{
  cv::Mat Q;
  double rms = cv::fisheye::stereoCalibrate(obj_points, left_points, right_points,
                                            K1_, D1_, K2_, D2_, img_size_,
                                            R_, T_, cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC);
  std::cout << "Calib error: " << rms << std::endl;
  int rect_flag = cv::fisheye::CALIB_ZERO_DISPARITY;
  cv::fisheye::stereoRectify(K1_, D1_, K2_, D2_, img_size_,
                              R_, T_, R1_, R2_, P1_, P2_, Q,
                              rect_flag, img_size_, 0, fov_);
}

void CalibrationModel::calibrateOmnidir(
  std::vector<std::vector<cv::Point2f>> left_points,
  std::vector<std::vector<cv::Point2f>> right_points,
  std::vector<std::vector<cv::Point3f>> obj_points)
{
  cv::Mat idx;
  std::vector<cv::Mat> rvecsL, tvecsL;
  cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, 0.0001);
  double rms = cv::omnidir::stereoCalibrate(obj_points, left_points, right_points, img_size_, img_size_,
                                            K1_, xi1_, D1_, K2_, xi2_, D2_,
                                            R_, T_, rvecsL, tvecsL, 0, criteria, idx);
  std::cout << "Calib error: " << rms << std::endl;
  cv::omnidir::stereoRectify(R_, T_, R1_, R2_);
  P1_ = (cv::Mat)cv::Matx33f(img_size_.width / fov_, 0, 0,
                              0, img_size_.height / fov_, 0,
                              0, 0, 1);
  P2_ = (cv::Mat)cv::Matx33f(img_size_.width / fov_, 0, 0,
                              0, img_size_.height / fov_, 0,
                              0, 0, 1);
}

void CalibrationModel::setParams(config::CalibrationModelParams cm_params)
{
  cam_type_ = cm_params.cam_type;
  img_size_ = cm_params.img_size;
  board_size_ = cm_params.board_size;
  square_size_ = cm_params.square_size;
  fov_ = cm_params.fov;
}

void CalibrationModel::calibrate(std::vector<cv::Mat> &left_imgs, std::vector<cv::Mat> &right_imgs)
{
  std::vector<std::vector<cv::Point2f>> left_points;
  std::vector<std::vector<cv::Point2f>> right_points;
  std::vector<std::vector<cv::Point3f>> obj_points;

  scanCheckerBoard(left_imgs, right_imgs, left_points, right_points);
  getObjPoints(left_points.size(), obj_points);
  std::cout << left_points.size() << " images scaned." << std::endl;

  switch (cam_type_)
  {
  case CameraType::PINHOLE:
    calibratePinhole(left_points, right_points, obj_points);
    break;
  case CameraType::FISHEYE:
    calibrateFisheye(left_points, right_points, obj_points);
    break;
  case CameraType::OMNIDIR:
    calibrateOmnidir(left_points, right_points, obj_points);
    break;
  default:
    throw std::runtime_error("CarlibrationModel: Invalid CameraType detected.");
    return;
  }

  is_calibrated_ = true;
}

void CalibrationModel::writeCameraParams(std::string path)
{
  if (!is_calibrated_)
      throw std::runtime_error("CarlibrationModel: Please calibrate befor write params.");

  cv::FileStorage fs(path, cv::FileStorage::WRITE);

  switch (cam_type_)
  {
  case CameraType::PINHOLE:
    fs << "camera_type" << "pinhole";
    fs << "fov" << 1.0;
    fs << "img_size" << img_size_;
    fs << "camera_params"
        << "{"
        << "K1" << K1_ << "K2" << K2_
        << "D1" << D1_ << "D2" << D2_
        << "R1" << R1_ << "R2" << R2_
        << "P1" << P1_ << "P2" << P2_
        << "R" << R_ << "T" << T_
        << "}";
    break;
  case CameraType::FISHEYE:
    fs << "camera_type" << "fisheye";
    fs << "fov" << fov_;
    fs << "img_size" << img_size_;
    fs << "camera_params"
        << "{"
        << "K1" << K1_ << "K2" << K2_
        << "D1" << D1_ << "D2" << D2_
        << "R1" << R1_ << "R2" << R2_
        << "P1" << P1_ << "P2" << P2_
        << "R" << R_ << "T" << T_
        << "}";
    break;
  case CameraType::OMNIDIR:
    fs << "camera_type" << "omnidir";
    fs << "fov" << fov_;
    fs << "img_size" << img_size_;
    fs << "camera_params"
        << "{"
        << "K1" << K1_ << "K2" << K2_
        << "D1" << D1_ << "D2" << D2_
        << "xi1" << xi1_ << "xi2" << xi2_
        << "R1" << R1_ << "R2" << R2_
        << "P1" << P1_ << "P2" << P2_
        << "R" << R_ << "T" << T_
        << "}";
    break;
  default:
    throw std::runtime_error("CarlibrationModel::writeCameraParams: Invalid CameraType detected.");
    return;
  }

  fs.release();

  std::cout << "Parameter saved. >> \"" << path << "\"" << std::endl;
}

} // namespace engine
