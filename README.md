# Fisheye-Depth-Estimation

3D reconstruction by fisheye stereo camera ([CaliCam® Fisheye Stereo Camera](https://astar.ai/products/stereo-camera))

Real-time & high quality fisheye stereo 3D reconstruction.


## Environment

* C++ 17
* OpneCV & OpenCV Contrib 4.7
* PointCloudLibrary 1.13
* Python 3.8

OpenCV should be compiled with [OpenEXR](https://openexr.com/en/latest/).


## Build & Run

### Build

```bash
sh build.sh
```

### Calibration

To remove camera distortion and stereo recitfication, calibrate camera with checker board.

```bash
build/calibration --config=<config file path> --images_dir==<calib images dir> --output=<calib data output path> --test_image=<test image path>
```

### Test 3D reconstruction

Before proceeding with real-time 3D reconstruction, verify the calibration results and parameters by testing a single-shot image.

```bash
build/test --config=<config file path> --calib=<calib data file path> --test_image=<test image path> --output_disp=<disparity image output path> --output_pcd=<point cloud output path>
```

### Real-time stereo

```bash
build/realtime_stereo --config=<config file path> --calib=<calib data file path>
```

Some configurations required for each step. See [config files](./configs/).


## Result

### Compute Disparity

**build/test**

* Original

![original](./samples/test.jpg "original")

* 3D preview ([Cloud Compare](https://www.danielgm.net/cc/))

![point_cloud](./samples/preview.webp "point_cloud")


## How does it work ?

Depth estimation with fisheye stereo camera requires some steps.

1. Stereo calibration
2. Image transformation
3. Disparity calculation
4. 3D reconstruction

### Stereo calibration

Firstly, to calculate the disparity between the images captured by two cameras, it's essential to correct the distortions inherent in fisheye cameras and align the parallel lines.

The required camera matrices can be acquired by photographing a specific checkerboard pattern from various angles and applying a corner detection algorithm.

* Checkerboard

![checkerboard](./samples/calib_sample.jpg "checkerboard")

### Image transformation

The image, now free from distortions and with aligned parallel lines, still exhibits a very small center due to the characteristics of the fisheye lens. 

The equirectangular transformation addresses this image area issue while preserving the alignment of parallel lines, improving the accuracy of the disparity calculation.

Equirectangular is a graphical method commonly used in world maps, where the axes correspond to latitude and longitude.

* Right after parallel lines aligned

![rectify](./samples/rect.png "rectify")

* After Equirectangular transformation

![eqrec](./samples/eqrec.png "eqrec")



### Disparity calculation
Disparity calculation is performed on the transformed images. A image matching algorithm can be used to calculate disparity. The matching algorithm cannot calculate disparity well for areas with little texture.

* Disparity

![disparity](./samples/disp.png "diaparity")

Once the disparity has been calculated, all that remains is to convert it to depth using a simple formula.

### 3D reconstruction

In the process of 3D reconstruction, we generate XYZRGB points from RGB images and disparity images.

The disparities derived from stereo matching of equirectangular images differ from the original disparities, so we convert them to the original disparities using coordinate mapping from equirectangular images to rectified images.

Similarly, we map the equirectangular RGB images to their recitfied coordinates before performing calculations. After the coordinate transformation, the method of creating a 3D point cloud from a standard image can be used.

* Image to Point Cloud

![cloud_compare](./samples/conversion.png)


## Key Features

### Disparity

The clarity near the center of rectified and equirectangular images greatly differs, significantly impacting the accuracy of stereo matching.　While high-precision stereo matching is possible with rectified images if the image is enlarged, it results in increased computational load and is inefficient compared to equirectangular images.

* Rectified vs Equirectangular

![comparison disparity](./samples/rect_eqrec_disparity.png)

### Transformation Map

The transformations from the original image to a rectified image, and from the rectified image to an equirectangular image, can both be represented by maps.

However, if we convert to an equirectangular image in the order of original image ⇒ rectified image ⇒ equirectangular image, the conversion to a rectified image causes the loss of central pixels, resulting in a coarse image.

Therefore, we combined the maps of the rectified conversion and the equirectangular conversion using cv::remap, making the transformation from the original image to an equirectangular image possible.

* High-quality equirectangular image

![high-quality equirectangular](./samples/remap.png)

## Future update

* CUDA-enabled stereo matching algorithm.
* Superpixel-based disparity refinement.
* Building a deep learning-based pipeline.


## For more detail

Detailed description of the algorithm here. (Japanese)

[Logic description (Qiita)](https://qiita.com/syunnsyunn74/items/155ee816f39691f021d2)

[Implement description (Qiita)](https://qiita.com/syunnsyunn74/items/6e248f7fbe87aa18e69d)