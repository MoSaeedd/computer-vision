#!/usr/local/opt/python@3.9/bin/python3.9 python3

import numpy as np
import cv2 
from matplotlib import pyplot as plt
from matplotlib import patches
np.set_printoptions(suppress=True)


# Read the stereo-pair of images
img_left = cv2.imread('I1_000000.png',cv2.IMREAD_COLOR)
img_right = cv2.imread('I2_000000.png',cv2.IMREAD_COLOR)

# Use matplotlib to display the two images
# _, image_cells = plt.subplots(2, 1, figsize=(10, 10))
# image_cells[0].imshow(img_left)
# image_cells[0].set_title('left image')
# image_cells[1].imshow(img_right)
# image_cells[1].set_title('right image')
#plt.show()

p_left =np.array([[6.452401e+02 , 0.000000e+00 , 6.359587e+02 , 0.000000e+00],[0.000000e+00 , 6.452401e+02 , 1.941291e+02 , 0.000000e+00],[0.000000e+00, 0.000000e+00 ,1.000000e+00 , 0.000000e+00]])

p_right= np.array([[6.452401e+02 , 0.000000e+00 , 6.359587e+02 , -3.682632e+02],[0.000000e+00 , 6.452401e+02 , 1.941291e+02 , 0.000000e+00],[0.000000e+00, 0.000000e+00 ,1.000000e+00 , 0.000000e+00]])

#p_left,_ = cv2.Rodrigues(p_left)
#p_left= cv2.UMat(p_left)
#p_right = cv2.UMat(p_right)
#p_right,_ = cv2.Rodrigues(p_right)


def compute_left_disparity_map(img_left, img_right):
    
    matcher = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    img_left= cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY) 
    img_right= cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
#    disp_left=matcher.compute(img_left, img_right)/16

    # Parameters
    num_disparities = 6*16
    block_size = 11
    
    min_disparity = 0
    window_size = 6
    
    # Stereo SGBM matcher
    left_matcher_SGBM = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute the left disparity map
    disp_left = cv2.normalize(left_matcher_SGBM.compute(img_left, img_right).astype(np.float32)/16,None, alpha=1, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
    disp_left[disp_left == 0] = 0.1
    # disp_left[disp_left == -1] = 0.1
    return disp_left


def calc_depth_map(disp_left, k_left, t_left, t_right):
    f= k_left[0,0]
    b= t_left[1]-t_right[1]
    depth_map= (f*b)/disp_left
    
    return depth_map


def decompose_projection_matrix(p):
    
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)    
    t = t / t[3]
    
    return k, r, t


# Compute the disparity map using the fuction above
disp_left = compute_left_disparity_map(img_left, img_right)
# Show the left disparity map
plt.figure(figsize=(10, 10))
plt.imshow(disp_left)
plt.show()

# Decompose each matrix
k_left, r_left, t_left = decompose_projection_matrix(p_left)
k_right, r_right, t_right = decompose_projection_matrix(p_right)

# Get the depth map by calling the above function
depth_map_left = calc_depth_map(disp_left, k_left, t_left, t_right)
# Display the depth map
plt.figure(figsize=(8, 8), dpi=100)
plt.imshow(depth_map_left, cmap='flag')
plt.show()




# print(cv2.minMaxLoc(depth_map_left))
# Display the matrices
# print("k_left \n", k_left)
# print("\nr_left \n", r_left)
# print("\nt_left \n", t_left)
# print("\nk_right \n", k_right)
# print("\nr_right \n", r_right)
# print("\nt_right \n", t_right)
