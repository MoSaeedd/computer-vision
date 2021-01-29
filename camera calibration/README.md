# Camera Calibration

### In this tutorial, the camera calibration process will be quickly explained.
### Goal:
* be able to map every object in 3d world coordinates to a pixel location in our image 
* obtain an image that correctly describe the scene as much as possible

## First, Why do we need to calibrate cameras?

There are several issues that we are trying to resolve when doing camera calibration

1. Undistortion: normally, cameras suffer from two types of distorsion, [radial](https://en.wikipedia.org/wiki/Distortion_(optics)#Radial_distortion) and tangential distorsion.

Radial distortion causes straight lines to appear curved. Radial distortion becomes larger the farther points are from the center of the image.
<figure>
<img src="https://upload.wikimedia.org/wikipedia/commons/6/63/Barrel_distortion.svg" height="200" width="200">
  <figcaption>Fig.1 - Barrel distortion. Source:Wikipedia.</figcaption>
</figure>


Tangential distortion occurs because the image-taking lense is not aligned perfectly parallel to the imaging plane. So, a perfectly alligned image will appear as skewed and some areas in the image may look nearer than expected.

![](https://www.researchgate.net/publication/332199146/figure/fig5/AS:743978198642690@1554389633677/Tangential-distortion.png)

Do you see how these types of distorsions can make some pixel locations map to false real world location ?
