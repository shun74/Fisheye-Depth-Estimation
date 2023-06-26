import cv2
from glob import glob
import sys
from Pipline import Pipeline
from FisheyeCalibrate import FisheyeCalibrator
from DepthEstimation import DepthEstimator

if __name__ == "__main__":
    image_dir = "calib_images/"
    images_path = glob(image_dir+"*.jpg")
    images = []
    for path in images_path:
        images.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))

    fisheyeCalibrator = FisheyeCalibrator()
    ret = fisheyeCalibrator.stereo_calibrate(
        images = images,
        image_size = (960, 1280),
        board = (7,10),
        square_size = 0.025,
        detail=False
    )
    baseline = -ret["tvec"][0][0]
    depthEstimator = DepthEstimator(
        num_disparities=100,
        window_size=7,
        baseline=baseline
    )
    test_dir = "images/"
    sample_image_path = test_dir + "test-1.jpg"
    if len(sys.argv) == 2:
        sample_image_path = test_dir + sys.argv[1]
    sample_image = cv2.imread(sample_image_path)
    sample_image = cv2.resize(sample_image, (2560, 960), interpolation=cv2.INTER_LINEAR)
    pipline = Pipeline(
        fisheyeCalibrator,
        depthEstimator,
        eqrec_mag_y=1.4,
        rec_show=True,
        eqrec_show=True,
        disp_show=True,
        threshold=(0.3,3.0),
        depth_show=True,
    )
    pipline.run(sample_image)
