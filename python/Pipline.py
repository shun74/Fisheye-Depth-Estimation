from FisheyeCalibrate import FisheyeCalibrator
from DepthEstimation import DepthEstimator

class Pipeline():
    def __init__(self, fisheyeCalibrator, depthEstimator, detail=False, rec_save=None, rec_show=False,eqrec_save=None, eqrec_show=False, eqrec_axis="vertical", eqrec_mag_x=1.0, eqrec_mag_y=1.0,disp_save=None, disp_show=False,threshold=(0.0,2.0), depth_save=None, depth_show=False):
        self._fisheyeCalibrator = fisheyeCalibrator
        self._fisheyeCalibrator.stereo_rectify(detail=detail)
        self._fisheyeCalibrator.create_rectify_map()
        self._fisheyeCalibrator.create_equirectangular_map(
            axis = eqrec_axis,
            mag_x = eqrec_mag_x,
            mag_y = eqrec_mag_y
        )
        self._depthEstimator = depthEstimator
        self._rec_save = rec_save
        self._rec_show = rec_show
        self._eqrec_save = eqrec_save
        self._eqrec_show = eqrec_show
        self._disp_save = disp_save
        self._disp_show = disp_show
        self._threshold = threshold
        self._depth_save = depth_save
        self._depth_show = depth_show

    def run(self, image):
        rectified_image = self._fisheyeCalibrator.get_rectified_image(
            image = image,
            save = self._rec_save,
            show = self._rec_show
        )
        eqrec_image = self._fisheyeCalibrator.get_equirectangular_image(
            image = rectified_image,
            save = self._eqrec_save,
            show = self._eqrec_show
        )
        self._depthEstimator.set_image(eqrec_image)
        self._depthEstimator.stereo_matching(
            save = self._disp_save,
            show = self._disp_show
        )
        self._depthEstimator.depth_estimation(
            threshold = self._threshold,
            save = self._depth_save,
            show = self._depth_show
        )
