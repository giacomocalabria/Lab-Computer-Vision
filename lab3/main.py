import cv2 as cv
import numpy as np

from HistEq import histeq

# Load the reference image
reference = cv.imread(".\dataset\data\peppa\peppa.png")

# Load the patches imagees
piece1 = cv.imread(".\dataset\data\peppa\pieces1.png")
piece2 = cv.imread(".\dataset\data\peppa\pieces2.png")

# Merge the two pieces images into one side by side
pieces = np.hstack((piece1, piece2))

# Create a window to display the reference image
cv.imshow('Reference', reference)
cv.imshow('Pieces', pieces)
cv.waitKey(0)
cv.destroyAllWindows()

# Segment the pieces that are in the pieces image

# 1. Convert the pieces image to grayscale
gray_pieces = cv.cvtColor(pieces, cv.COLOR_BGR2GRAY)

# 2. Apply a median filter in order to remove white noise on the puzzle pieces
gray_pieces = cv.medianBlur(gray_pieces, 5)

# 3. Apply a binary threshold to segment the pieces
_, thresh = cv.threshold(gray_pieces, 250, 255, cv.THRESH_BINARY_INV)

# 4. Apply a blur filter to the thresholded image
thresh = cv.blur(thresh, (3,3))

# 5. Find the contours of the pieces
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 6. Draw the contours on the pieces image
cv.drawContours(pieces, contours, -1, (0,255,0), 3)

# 7. Create a window to display the segmented pieces
cv.imshow('Segmented Pieces', pieces)
cv.waitKey(0)
cv.destroyAllWindows()
