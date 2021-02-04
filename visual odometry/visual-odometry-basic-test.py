import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
img1= cv2.imread('1.png',0)
img2= cv2.imread('2.png',0)

orb = cv2.ORB_create()
kp1=orb.detect(img1)
kp2=orb.detect(img2)
kp1, des1 = orb.compute(img1,kp1)
kp2, des2 = orb.compute(img2,kp2)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:40], None,flags=2)

plt.imshow(img3),plt.show()