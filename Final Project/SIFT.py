import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

image = cv.imread("dataset/train/images/road55.png")

if image is None:
    print("Could not read the image.")
    exit()

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#gray = cv.GaussianBlur(gray, (9, 9), 1)

sift = cv.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(gray, None)

image_with_keypoints = cv.drawKeypoints(gray, keypoints, None)

plt.imshow(image_with_keypoints)
plt.axis('off')
plt.show()