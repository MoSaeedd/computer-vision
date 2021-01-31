import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from sklearn.preprocessing import normalize

DPI=96
DATASET = "data/1"
DATASET_LEFT = DATASET+"/left/"
DATASET_RIGHT = DATASET+"/right/"
DATASET_DISPARITIES = DATASET+"/disparities/"
DATASET_COMBINED = DATASET+"/combined/"
DATASET_DEPTH = DATASET+"/depth/"
p_left =np.array([[6.452401e+02 , 0.000000e+00 , 6.359587e+02 , 0.000000e+00],[0.000000e+00 , 6.452401e+02 , 1.941291e+02 , 0.000000e+00],[0.000000e+00, 0.000000e+00 ,1.000000e+00 , 0.000000e+00]])

p_right= np.array([[6.452401e+02 , 0.000000e+00 , 6.359587e+02 , -3.682632e+02],[0.000000e+00 , 6.452401e+02 , 1.941291e+02 , 0.000000e+00],[0.000000e+00, 0.000000e+00 ,1.000000e+00 , 0.000000e+00]])


def calc_depth_map(disp_left, k_left, t_left, t_right, name):
    f= k_left[0,0]
    b= t_left[1]-t_right[1]
    depth_map= (f*b)/disp_left
    fig = plt.figure(figsize=(depth_map.shape[1]/DPI, depth_map.shape[0]/DPI), dpi=DPI, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(depth_map)
    plt.savefig(DATASET_DEPTH+name)
    plt.close()
    return depth_map


def decompose_projection_matrix(p):

    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)    
    t = t / t[3]

    return k, r, t

k_left, r_left, t_left = decompose_projection_matrix(p_left)
k_right, r_right, t_right = decompose_projection_matrix(p_right)


def process_frame(left, right, name):
    kernel_size = 3
    smooth_left = cv2.GaussianBlur(left, (kernel_size,kernel_size), 1.5)
    smooth_right = cv2.GaussianBlur(right, (kernel_size, kernel_size), 1.5)

    window_size = 9    
    left_matcher = cv2.StereoSGBM_create(
        numDisparities=96,
        blockSize=7,
        P1=8*3*window_size**2,
        P2=32*3*window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=16,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(80000)
    wls_filter.setSigmaColor(1.2)

    disparity_left = np.int16(left_matcher.compute(smooth_left, smooth_right))
    disparity_right = np.int16(right_matcher.compute(smooth_right, smooth_left) )

    wls_image = wls_filter.filter(disparity_left, smooth_left, None, disparity_right)
    wls_image = cv2.normalize(src=wls_image, dst=wls_image, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    wls_image = np.uint8(wls_image)

    fig = plt.figure(figsize=(wls_image.shape[1]/DPI, wls_image.shape[0]/DPI), dpi=DPI, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(wls_image, cmap='jet')
    plt.savefig(DATASET_DISPARITIES+name)
    plt.close()
    calc_depth_map(disparity_left, k_left, t_left, t_right,name)
    create_combined_output(left, right, name)



def create_combined_output(left, right, name):
	combined = np.concatenate((left, right, cv2.imread(DATASET_DISPARITIES+name),cv2.imread(DATASET_DEPTH+name)), axis=0)
	cv2.imwrite(DATASET_COMBINED+name, combined)

def process_dataset():
	left_images = [f for f in os.listdir(DATASET_LEFT) if not f.startswith('.')]
	right_images = [f for f in os.listdir(DATASET_RIGHT) if not f.startswith('.')]
	assert(len(left_images)==len(right_images))
	left_images.sort()
	right_images.sort()
	for i in range(len(left_images)):
		left_image_path = DATASET_LEFT+left_images[i]
		right_image_path = DATASET_RIGHT+right_images[i]
		left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
		right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)
		process_frame(left_image, right_image, left_images[i])

if __name__== "__main__":
	process_dataset()
