import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class DepthEstimator():
    def __init__(self, num_disparities, window_size, baseline):
        self._num_disparities = num_disparities # disparity detection pixel range
        self._window_size = window_size         # matching window size
        self._baseline = baseline               # baseline length in meter
        self._image = None                      # image
        self._left_image = None                 # image left half
        self._right_image = None                # image right half
        self._disparity_map = None              # disparity map
        self._depth_map = None                  # depth map

    def set_image(self, image):
        print("set image")
        self._image = image
        self._left_image = self._image[:,:self._image.shape[1]//2]
        self._right_image = self._image[:,self._image.shape[1]//2:]

    def stereo_matching(self, save=None, show=True):
        # image -> disparity map
        assert self._image is not None, "Plz set image first."
        print("stereo matching ...")
        stereo = cv2.StereoSGBM_create(
            numDisparities = self._num_disparities,
            blockSize = self._window_size,
            P1 = 8 * 3 * self._window_size**2,
            P2 = 32 * 3 * self._window_size**2,
            disp12MaxDiff = 1,
            uniquenessRatio = 10,
            speckleWindowSize = 150,
            speckleRange = 32,
            mode=cv2.StereoSGBM_MODE_HH
        )
        blur = 7
        self._left_image = cv2.GaussianBlur(self._left_image,(blur,blur),blur)
        self._right_image = cv2.GaussianBlur(self._right_image,(blur,blur),blur)
        self._disparity_map = stereo.compute(self._left_image, self._right_image).astype(np.float32)
        self._disparity_map = self._disparity_map / 16.0
        if save:
            fig = plt.figure(figsize=(15,15))
            plt.imshow(self._disparity_map)
            plt.axis("off")
            fig.savefig(save)
        if show:
            self._show_images(images=[cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB), self._disparity_map], titles=["Original", "Disparity"], figsize=(30,20), subplot=(1,2))
        return self._disparity_map

    def depth_estimation(self, axis="vertical",threshold=(0.0,2.0), save=None, show=True):
        assert self._disparity_map is not None, "Plz run stereo matching first."
        height = self._left_image.shape[0]
        width = self._left_image.shape[1]
        self._depth_map = np.zeros(self._disparity_map.shape , np.float32)
        if axis == "vertical":
            for y in range(height):
                for x in range(width):
                    lamb = (1.0 - y/(height/2.0)) * (math.pi/2.0)
                    phi = (x/(width/2.0) - 1.0) * (math.pi/2.0)
                    delta_phi = self._disparity_map[y,x] / (width/2.0) * (math.pi/2.0)
                    distance = self._baseline * math.sin(math.pi/2.0 - phi) / (1e-7+math.sin(delta_phi) * math.cos(lamb))
                    self._depth_map[y,x] = distance *math.cos(phi) * math.cos(lamb)
        elif axis == "horizontal":
            for y in range(height):
                for x in range(width):
                    lamb = (1.0 + y/(height/2.0)) * (math.pi/2.0)
                    phi = (1.0 - x/(width/2.0)) * (math.pi/2.0)
                    delta_lamb = -(self._disparity_map[y,x] / (width/2.0)) * (math.pi/2.0)
                    distance = self._baseline * math.sin(math.pi/2.0 - lamb) / (1e-7+math.sin(delta_lamb) * math.cos(phi))
                    self._depth_map[y,x] = distance * math.cos(phi)
        self._depth_map = np.where((self._depth_map < threshold[0]) | (self._depth_map > threshold[1]), 0.0, self._depth_map)
        if save:
            fig = plt.figure(figsize=(15,15))
            plt.imshow(self._depth_map)
            plt.axis("off")
            fig.savefig(save)
        if show:
            self._show_images(images=[cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB), self._depth_map], titles=["Original", "Depth"], figsize=(30,20), subplot=(1,2),color_bar=True)
        return self._depth_map

    def _show_images(self, images, titles, figsize, subplot, color_bar=False):
        # images -> show images
        plt.figure(figsize=figsize)
        for i, (image, title) in enumerate(zip(images, titles)):
            plt.subplot(subplot[0], subplot[1], i+1)
            plt.imshow(image)
            plt.title(title)
            plt.axis("off")
            if i==1 and color_bar:
                plt.colorbar()
        plt.show()