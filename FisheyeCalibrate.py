import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

class FisheyeCalibrator():
    def __init__(self):
        self._image_size = None             # Image size                        : (height, width)
        self._board = None                  # Baord shape                       : (rows, cols)
        self._square_size = None            # Board square size in meter        : float
        self._ret = None                    # Stereo calibration return         : float
        self._left_K = np.zeros((3,3))      # Left camera matrix                : 3*3
        self._right_K = np.zeros((3,3))     # Right camera matrix               : 3*3
        self._left_D = np.zeros((4,1))      # Left camera distortion vector     : 4*1
        self._right_D = np.zeros((4,1))     # Right camera distortion vector    : 4*1
        self._rvec = None                   # cameras rotation vector           : 1*1*3
        self._tvec = None                   # cameras translation vector        : 1*1*3
        self._left_R = np.zeros((3,3))      # Left camera rotation matrix       : 3*3
        self._left_P = np.zeros((3,4))      # Left camera projection matrix     : 3*4
        self._right_R = np.zeros((3,3))     # Right camera rotation matrix      : 3*3
        self._right_P = np.zeros((3,4))     # Right camera projection matrix    : 3*4
        self._Q = np.zeros((4,4))           # Disparity-to-depth mapping matrix : 4*4
        self._left_rectify_map_x = None     # Left rectify lookup table x       : height*width
        self._left_rectify_map_y = None     # Left rectify lookup table y       : height*width   
        self._right_rectify_map_x = None    # Right rectify lookup table x      : height*width
        self._right_rectify_map_y = None    # Right rectify lookup table y      : height*width
        self._left_eqrec_map_x = None       # Left eqrec lookup table x         : height*width
        self._left_eqrec_map_y = None       # Left eqrec lookup table y         : height*width   
        self._right_eqrec_map_x = None      # Right eqrec lookup table x        : height*width
        self._right_eqrec_map_y = None      # Right eqrec lookup table y        : height*width
        self._mag = 2.0                     # Fov related prams in rectify      : float

    def stereo_calibrate(self, images, image_size, board=(6,9), square_size=0.027, detail=False):
        # images -> calibrate params(left K, right K, left D, right D)
        self._image_size = image_size[::-1]
        self._board = board
        self._square_size = square_size
        print(f"Board type: rows{board[0]}, cols{board[1]}")
        assert isinstance(images, list), "images must be list"
        # Corners detection
        print("detecting corners ...")
        left_corners = []
        right_corners = []
        object_point = self._create_object_point()
        object_points = []
        for image in tqdm(images):
            left_corner, right_corner = self._detect_corners(image)
            # All corners must be detected. n_corners = (rows * cols)
            assert len(left_corner)==len(right_corner)==board[0]*board[1], "Corners detection failed."
            left_corners.append(left_corner.reshape(1,-1,2))
            right_corners.append(right_corner.reshape(1,-1,2))
            object_points.append(object_point)
        print("corners detection was successful.")
        print("stereo calibrating ...")
        self._rvec = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(images))]
        self._tvec = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(images))]
        self._ret, self._left_K, self._left_D, self._right_K, self._right_D, self._rvec, self._tvec = cv2.fisheye.stereoCalibrate(
            objectPoints=object_points,
            imagePoints1=left_corners,
            imagePoints2=right_corners,
            K1=self._left_K,
            D1=self._left_D,
            K2=self._right_K,
            D2=self._right_D,
            imageSize=self._image_size,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        )
        print("stereo calibration was successful.")
        if detail:
            self.show_stereo_params()
        stereo_params_dict = {
            "left_K": self._left_K,
            "right_K": self._right_K,
            "left_D": self._left_D,
            "right_D": self._right_D,
            "rvec": self._rvec,
            "tvec": self._tvec
        }
        return stereo_params_dict

    def stereo_rectify(self, detail=False):
        # _ -> rectify params(left R, right R, left P, right R, Q)
        print("stereo rectify...")
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        self._left_R, self._right_R, self._left_P, self._right_P, self._Q = cv2.fisheye.stereoRectify(
            K1=self._left_K,
            D1=self._left_D,
            K2=self._right_K,
            D2=self._right_D,
            imageSize=self._image_size,
            R=self._rvec,
            tvec=self._tvec,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
            newImageSize=(int(self._image_size[0]*self._mag), int(self._image_size[1]*self._mag)),
            fov_scale=self._mag
        )
        print("stereo rectification was successful.")
        if detail:
            self.show_rectify_params()
        rectify_params_dict = {
            "left_R": self._left_R,
            "right_R": self._right_R,
            "left_P": self._left_P,
            "right_P": self._right_P,
            "Q": self._Q
        }
        return rectify_params_dict

    def create_rectify_map(self):
        # _ -> rectify maps(left x, left y, right x, right y)
        self._left_rectify_map_x, self._left_rectify_map_y = cv2.fisheye.initUndistortRectifyMap(
            K=self._left_K,
            D=self._left_D,
            R=self._left_R,
            P=self._left_P,
            size=(int(self._image_size[0]*self._mag), int(self._image_size[1]*self._mag)),
            # size=self._image_size,
            m1type=cv2.CV_32FC1
        )
        self._right_rectify_map_x, self._right_rectify_map_y = cv2.fisheye.initUndistortRectifyMap(
            K=self._right_K,
            D=self._right_D,
            R=self._right_R,
            P=self._right_P,
            size=(int(self._image_size[0]*self._mag), int(self._image_size[1]*self._mag)),
            # size=self._image_size,
            m1type=cv2.CV_32FC1
        )
        print("rectify lookup table ceraeted.")
        rectify_map_dict = {
            "left_map_x": self._left_rectify_map_x,
            "left_map_y": self._left_rectify_map_y,
            "right_map_x": self._right_rectify_map_x,
            "right_map_y": self._right_rectify_map_y
        }
        return rectify_map_dict

    def create_equirectangular_map(self, axis="vertical", mag_x=1.0, mag_y=1.0):
        # _ -> eqrec maps(left x, left y, right x, right y)
        assert axis in ["horizontal", "vertical"], "Select horizontal or vertical as axis."
        w,h = self._image_size
        w = int(w * mag_x)
        h = int(h * mag_y)
        self._left_eqrec_map_x = np.zeros((h,w), dtype=np.float32)
        self._left_eqrec_map_y = np.zeros((h,w), dtype=np.float32)
        self._right_eqrec_map_x = np.zeros((h,w), dtype=np.float32)
        self._right_eqrec_map_y = np.zeros((h,w), dtype=np.float32)
        # left map
        fx = self._left_P[0,0]
        fy = self._left_P[1,1]
        cx = self._left_P[0,2]
        cy = self._left_P[1,2]
        for y in range(h):
            for x in range(w):
                if axis=="vertical":
                    lamb = (1.0 - y/(h/2.0)) * (math.pi/2.0)
                    phi = (x/(w/2.0) - 1.0) * (math.pi/2.0)
                    vs_y = math.tan(lamb)
                    vs_x = math.tan(phi) / math.cos(lamb)
                    rec_x = cx + vs_x*fx
                    rec_y = cy - vs_y*fy
                    self._left_eqrec_map_x[y,x] = rec_x
                    self._left_eqrec_map_y[y,x] = rec_y
                elif axis=="horizontal":
                    lamb = (1.0 - x/(w/2.0)) * (math.pi/2.0)
                    phi = (1.0 + y/(h/2.0)) * (math.pi/2.0)
                    vs_x = math.tan(lamb)
                    vs_y = math.tan(phi) / math.cos(lamb)
                    rec_x = cx - vs_x*fx
                    rec_y = cy + vs_y*fy
                    self._left_eqrec_map_x[y,x] = rec_x
                    self._left_eqrec_map_y[y,x] = rec_y
        # right map
        fx = self._right_P[0,0]
        fy = self._right_P[1,1]
        cx = self._right_P[0,2]
        cy = self._right_P[1,2]
        for y in range(h):
            for x in range(w):
                if axis=="vertical":
                    lamb = (1.0 - y/(h/2.0)) * (math.pi/2.0)
                    phi = (x/(w/2.0) - 1.0) * (math.pi/2.0)
                    vs_y = math.tan(lamb)
                    vs_x = math.tan(phi) / math.cos(lamb)
                    rec_x = cx + vs_x*fx
                    rec_y = cy - vs_y*fy
                    self._right_eqrec_map_x[y,x] = rec_x
                    self._right_eqrec_map_y[y,x] = rec_y
                elif axis=="horizontal":
                    lamb = (1.0 - x/(w/2.0)) * (math.pi/2.0)
                    phi = (1.0 + y/(h/2.0)) * (math.pi/2.0)
                    vs_x = math.tan(lamb)
                    vs_y = math.tan(phi) / math.cos(lamb)
                    rec_x = cx - vs_x*fx
                    rec_y = cy + vs_y*fy
                    self._right_eqrec_map_x[y,x] = rec_x
                    self._right_eqrec_map_y[y,x] = rec_y
        print("equirectangular lookup table ceraeted.")
        eqrec_map_dict = {
            "left_map_x": self._left_eqrec_map_x,
            "left_map_y": self._left_eqrec_map_y,
            "right_map_x": self._right_eqrec_map_x,
            "right_map_y": self._right_eqrec_map_y
        }
        return eqrec_map_dict 

    def get_rectified_image(self, image, save=None, show=True):
        # image -> rectified_image
        msg = "Plz create map first."
        assert self._left_rectify_map_x is not None, msg
        assert self._left_rectify_map_y is not None, msg
        assert self._right_rectify_map_x is not None, msg
        assert self._right_rectify_map_y is not None, msg
        left_image, right_image = self._split_image(image)
        left_rectified_image = cv2.remap(left_image, self._left_rectify_map_x, self._left_rectify_map_y, cv2.INTER_LINEAR)
        right_rectified_image = cv2.remap(right_image, self._right_rectify_map_x, self._right_rectify_map_y, cv2.INTER_LINEAR)
        rectified_image = np.concatenate([left_rectified_image, right_rectified_image], axis=1)
        if save is not None:
            cv2.imwrite(save, rectified_image)
        if show:
            self._show_images(images=[image, rectified_image], titles=["Original", "Rectified"], figsize=(30,20), subplot=(2,1))
        return rectified_image

    def get_equirectangular_image(self, image, save=None, show=True):
        # image -> rectified_image
        msg = "Plz create map first."
        assert self._left_eqrec_map_x is not None, msg
        assert self._left_eqrec_map_y is not None, msg
        assert self._right_eqrec_map_x is not None, msg
        assert self._right_eqrec_map_y is not None, msg
        left_image, right_image = self._split_image(image)
        left_eqrec_image = cv2.remap(left_image, self._left_eqrec_map_x, self._left_eqrec_map_y, cv2.INTER_LINEAR)
        right_eqrec_image = cv2.remap(right_image, self._right_eqrec_map_x, self._right_eqrec_map_y, cv2.INTER_LINEAR)
        eqrec_image = np.concatenate([left_eqrec_image, right_eqrec_image], axis=1)
        if save is not None:
            cv2.imwrite(save, eqrec_image)
        if show:
            self._show_images(images=[image, eqrec_image], titles=["Original", "Equirectangular"], figsize=(30,20), subplot=(2,1))
        return eqrec_image

    def show_stereo_params(self):
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print("stereo params")
        print("###################################")
        print(f"Return: {self._ret}")
        print(f"Left K:\n{self._left_K}")
        print(f"Left D:\n{self._left_D}")
        print(f"Right K:\n{self._right_K}")
        print(f"Right D:\n{self._right_D}")
        print(f"R vector:\n{self._rvec}")
        print(f"T vector:\n{self._tvec}")
        print("###################################")

    def show_rectify_params(self):
        print("rectify params")
        print("###################################")
        print("Left R:\n",self._left_R)
        print("Right R:\n",self._right_R)
        print("Left P:\n",self._left_P)
        print("Right P:\n",self._right_P)
        print("Q:\n",self._Q)
        print("###################################")

    def _split_image(self, image):
        # image -> left_half, right_half
        left_image = image[:,:image.shape[1]//2]
        right_image = image[:, image.shape[1]//2:]
        return left_image, right_image

    def _create_object_point(self):
        # _ -> object points grid refered board
        object_point = np.zeros((1, self._board[0]*self._board[1], 3), np.float32)
        object_point[0,:,:2] = np.mgrid[0:self._board[0], 0:self._board[1]].T.reshape(-1, 2)
        object_point = object_point*self._square_size 
        return object_point

    def _detect_corners(self, image):
        # image -> left half corners point, right corners point
        left_image, right_image = self._split_image(image)
        flag = cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        left_ret, left_corner = cv2.findChessboardCorners(left_image, self._board, flag)
        if len(left_corner)>0:
            for points in left_corner:
                cv2.cornerSubPix(left_image, points, winSize=(3, 3), zeroZone=(-1,-1), criteria=criteria)
        right_ret, right_corner = cv2.findChessboardCorners(right_image, self._board, flag)
        if len(right_corner)>0:
            for points in right_corner:
                cv2.cornerSubPix(right_image, points, winSize=(3, 3), zeroZone=(-1,-1), criteria=criteria)
        return left_corner, right_corner

    def _show_images(self, images, titles, figsize, subplot):
        # images -> show images
        plt.figure(figsize=figsize)
        for i, (image, title) in enumerate(zip(images, titles)):
            plt.subplot(subplot[0], subplot[1], i+1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis("off")
        plt.show()