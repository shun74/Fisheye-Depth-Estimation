import cv2
from glob import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from FisheyeCalibrate import FisheyeCalibrator
from DepthEstimation import DepthEstimator

if __name__ == "__main__":
    image_dir = "./images/"
    images_path = glob(image_dir + "checker_*.png")
    images = []
    for path in images_path:
        images.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))
    fisheyeCalibrator = FisheyeCalibrator()
    ret = fisheyeCalibrator.stereo_calibrate(
        images = images,
        image_size = (960, 1280),
        board = (6,9),
        square_size = 0.04,
    )
    baseline = -ret["tvec"][0][0]
    rec_params_dict = fisheyeCalibrator.stereo_rectify()
    fisheyeCalibrator.create_rectify_map()
    fisheyeCalibrator.create_equirectangular_map(
        mag_y = 1.4
    )

    sample_image_path = image_dir + "test1.png"
    if len(sys.argv) == 2:
        sample_image_path = image_dir + sys.argv[1]
    sample_image = cv2.imread(sample_image_path)

    rectified_image = fisheyeCalibrator.get_rectified_image(
        image = sample_image,
        show=False,
        save = "./images/rec.png"
    )
    eqrec_image = fisheyeCalibrator.get_equirectangular_image(
        image = rectified_image,
        show=False,
        save = "./images/eqrec.png"
    )
    depthEstimator = DepthEstimator(
        num_disparities=100,
        window_size=7,
        baseline=baseline
    )
    depthEstimator.set_image(eqrec_image)
    depthEstimator.stereo_matching(
        show=False,
        save = "./images/disp.png",
    )
    depthEstimator.depth_estimation(
        threshold=(0.3,3.0),
        show=False,
        save = "./images/depth.png"
    )
