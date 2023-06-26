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

1. Build
```bash
sh build.sh
```

2. Calibration
```bash
./build/calibration
```

3. Realtime stereo
```bash
./build/realtime_stereo
```

## Result

### Compute Disparity
* Original
![original](./images/test-1.jpg "original")

* Disparity (Aligned)
![disparity](./images/disp.png "disparity")


### Realtime 3D viewer
* 3D Point Cloud
![point_cloud1](./images/pcd-1.png "point_cloud1")
![point_cloud2](./images/pcd-2.png "point_cloud2")

## How does it work ?
Depth estimation with fisheye stereo camera requires some steps.
1. Stereo calibration
2. Image transformation
3. Disparity calculation
4. 3D reconstruction

### Stereo calibration
First, in order to calculate the disparity between the two camera images, it is necessary to remove the distortions in fisheye cameras and align the parallel lines. 

 The necessary camera matrices can be obtained by photographing a particular checkerboard from various angles and applying a corner detection algorithm.

* Checkerboard
![checkerboard](./images/calib_sample.jpg "checkerboard")

### Image transformation
The image is now clear of distortions and the parallel lines are aligned, but the center of the image is very small due to the characteristics of the fisheye lens. 

Equirectangular transformation improves the accuracy of the disparity calculation by solving the image area problem while maintaining the parallel lines.

Equirectangular is a graphical method used in world maps, where the axes are latitude and longitude.

* Right after parallel lines aligned
![rectify](./images/rect.png "rectify")

* After Equirectangular transformation
![eqrec](./images/eqrec.png "eqrec")

### Disparity calculation
Disparity calculation is performed on the transformed images. A image matching algorithm can be used to calculate disparity. The matching algorithm cannot calculate disparity well for areas with little texture.

* Disparity
![disparity](./images/disp.png "diaparity")

Once the disparity has been calculated, all that remains is to convert it to depth using a simple formula.

### 3D reconstruction



## Future update

* CUDA-enabled stereo matching algorithm.
* Superpixel-based disparity refinement.
* Building a deep learning-based pipeline.

## For more detail

Detailed description of the algorithm here. (Japanese)

[Logic description (Qiita)](https://qiita.com/syunnsyunn74/items/155ee816f39691f021d2)

[Implement description (Qiita)](https://qiita.com/syunnsyunn74/items/6e248f7fbe87aa18e69d)