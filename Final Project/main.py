import cv2 as cv
import numpy as np

from HistEq import histeq

# Load the reference image
reference = cv.imread(".\dataset\data\peppa\peppa.png")

# Load the patches imagees
piece1 = cv.imread(".\dataset\data\peppa\pieces1.png")
piece2 = cv.imread(".\dataset\data\peppa\pieces2.png")

print(reference.shape)
print(piece1.shape)
print(piece2.shape)

# Merge the two pieces images into one side by side
pieces = np.hstack((piece1, piece2))

# Create a window to display the reference image
cv.imshow('Reference', reference)
cv.imshow('Pieces', pieces)
cv.waitKey(0)
cv.destroyAllWindows()