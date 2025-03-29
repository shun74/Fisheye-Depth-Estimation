
#include <filesystem>

#include <opencv2/opencv.hpp>

#include <calibration_model.h>

void loadStereoImages(std::string img_dir, std::vector<cv::Mat> &left_imgs, std::vector<cv::Mat> &right_imgs)
{
    std::filesystem::path dir = img_dir;
    for (const auto &entry : std::filesystem::directory_iterator(dir))
    {
        if (entry.is_regular_file())
        {
            cv::Mat image = cv::imread(entry.path(), cv::IMREAD_GRAYSCALE);
            if (!image.empty())
            {
                left_imgs.push_back(image(cv::Rect(0, 0, image.cols / 2, image.rows)));
                right_imgs.push_back(image(cv::Rect(image.cols / 2, 0, image.cols / 2, image.rows)));
            }
        }
    }
}

class CalibrationModelImpl : public CalibrationModel
{
  private:
    void scanCheckerBoard(std::vector<cv::Mat> &left_imgs, std::vector<cv::Mat> &right_imgs,
                          std::vector<std::vector<cv::Point2f>> &left_points,
                          std::vector<std::vector<cv::Point2f>> &right_points)
    {
        int grid_num = board_size_.width * board_size_.height;
        for (int i = 0; i < static_cast<int>(left_imgs.size()); i++)
        {
            std::vector<cv::Point2f> left_corners;
            std::vector<cv::Point2f> right_corners;
            bool found1 = cv::findChessboardCorners(left_imgs[i], board_size_, left_corners,
                                                    cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
            bool found2 = cv::findChessboardCorners(right_imgs[i], board_size_, right_corners,
                                                    cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
            if (found1 && found2 && static_cast<int>(left_corners.size()) == grid_num &&
                static_cast<int>(right_corners.size()) == grid_num)
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

    void getObjPoints(int num, std::vector<std::vector<cv::Point3f>> &obj_points)
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

    void calibrateFisheye(std::vector<std::vector<cv::Point2f>> &left_points,
                          std::vector<std::vector<cv::Point2f>> &right_points,
                          std::vector<std::vector<cv::Point3f>> &obj_points)
    {
        cv::Mat Q;
        double rms = cv::fisheye::stereoCalibrate(obj_points, left_points, right_points, K1_, D1_, K2_, D2_, img_size_,
                                                  R_, T_, cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC);
        std::cout << "Calib error: " << rms << std::endl;
        int rect_flag = cv::fisheye::CALIB_ZERO_DISPARITY;
        cv::fisheye::stereoRectify(K1_, D1_, K2_, D2_, img_size_, R_, T_, R1_, R2_, P1_, P2_, Q, rect_flag, img_size_,
                                   0, fov_);
    }

  public:
    explicit CalibrationModelImpl(cv::Size img_size, cv::Size board_size, float square_size, float fov)
        : img_size_(img_size), board_size_(board_size), square_size_(square_size), fov_(fov), is_calibrated_(false)
    {
        K1_ = cv::Mat::ones(3, 3, CV_64F);
        K2_ = cv::Mat::ones(3, 3, CV_64F);
        D1_ = cv::Mat::ones(4, 1, CV_64F);
        D2_ = cv::Mat::ones(4, 1, CV_64F);
        R_ = cv::Mat::ones(3, 3, CV_64F);
        T_ = cv::Mat::ones(3, 1, CV_64F);

        K1_.at<double>(0, 0) = K2_.at<double>(0, 0) = img_size.width * 0.3;
        K1_.at<double>(1, 1) = K2_.at<double>(1, 1) = img_size.width * 0.3;
        K1_.at<double>(0, 2) = img_size.width / 2.0;
        K2_.at<double>(0, 2) = img_size.width / 2.0;
        K1_.at<double>(1, 2) = img_size.height / 2.0;
        K2_.at<double>(1, 2) = img_size.height / 2.0;
    }

    void calibrate(std::vector<cv::Mat> &left_imgs, std::vector<cv::Mat> &right_imgs) override
    {
        std::vector<std::vector<cv::Point2f>> left_points;
        std::vector<std::vector<cv::Point2f>> right_points;
        std::vector<std::vector<cv::Point3f>> obj_points;

        scanCheckerBoard(left_imgs, right_imgs, left_points, right_points);
        getObjPoints(left_points.size(), obj_points);
        std::cout << left_points.size() << " images scanned." << std::endl;

        if (left_points.empty())
        {
            throw std::runtime_error("No valid calibration images found");
        }

        calibrateFisheye(left_points, right_points, obj_points);

        is_calibrated_ = true;
    }

    void writeCameraParams(std::string path) override
    {
        if (!is_calibrated_)
            throw std::runtime_error("CalibrationModel: Please calibrate before writing parameters.");

        cv::FileStorage fs(path, cv::FileStorage::WRITE);

        fs << "camera_type" << "fisheye";
        fs << "fov" << fov_;
        fs << "img_size" << img_size_;
        fs << "camera_params"
           << "{"
           << "K1" << K1_ << "K2" << K2_ << "D1" << D1_ << "D2" << D2_ << "R1" << R1_ << "R2" << R2_ << "P1" << P1_
           << "P2" << P2_ << "R" << R_ << "T" << T_ << "}";

        fs.release();

        std::cout << "Parameters saved to \"" << path << "\"" << std::endl;
    }

  private:
    cv::Mat K1_, K2_, D1_, D2_, R1_, R2_, P1_, P2_, R_, T_;
    cv::Size img_size_;
    cv::Size board_size_;
    float square_size_;
    float fov_;
    bool is_calibrated_;
};

std::unique_ptr<CalibrationModel> CalibrationModel::create(cv::Size img_size, cv::Size board_size, float square_size,
                                                           float fov)
{
    return std::make_unique<CalibrationModelImpl>(img_size, board_size, square_size, fov);
}
