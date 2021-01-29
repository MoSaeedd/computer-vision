# Camera Calibration

### In this tutorial, the camera calibration process will be quickly explained.
### Goal:
* be able to map every object in 3d world coordinates to a pixel location in our image 
* obtain an image that correctly describe the scene as much as possible

## First, Why do we need to calibrate cameras?

There are several issues that we are trying to resolve when doing camera calibration

Undistortion: normally, cameras suffer from two types of distorsion, [radial](https://en.wikipedia.org/wiki/Distortion_(optics)#Radial_distortion) and tangential distorsion.

  Radial distortion causes straight lines to appear curved. Radial distortion becomes larger the farther points are from the center of the image.
  <figure>
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/63/Barrel_distortion.svg" height="200" width="200">
    <figcaption>Fig.1 - Barrel distortion. Source:Wikipedia.</figcaption>
  </figure>


  Tangential distortion occurs because the image-taking lense is not aligned perfectly parallel to the imaging plane. So, a perfectly alligned image will appear as skewed and some areas in the image may look nearer than expected.

  <figure>
  <img src="https://www.researchgate.net/publication/332199146/figure/fig5/AS:743978198642690@1554389633677/Tangential-distortion.png" >
    <figcaption>Fig.2 - Tangantial distortion. Source:researchgate.</figcaption>
  </figure>

## Do you see how these types of distorsions can make some pixel locations map to false real world location ?

Not only distorsion, in order to correctly map world points to pixel locations, we need to account for other camera intensic and Extrensic parameters. Intrinsic parameters are specific to a camera. They include information like focal length ( fx,fy) and optical centers ( cx,cy). The focal length and optical centers can be used to create a camera matrix, which can be used to remove distortion due to the lenses of a specific camera. The camera matrix is unique to a specific camera, so once calculated, it can be reused on other images taken by the same camera. It is expressed as a 3x3 matrix:

                                                                    [[fx 0 cx],
                                                                     [0 fy cy],
                                                                     [0 0 0 1]]
                                                                             
 Extrinsic parameters corresponds to rotation and translation vectors which translates a coordinates of a 3D point to a coordinate system.



sources:

https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
