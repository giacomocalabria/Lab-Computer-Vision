import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from functions.HistEq import histeq

from functions.SeparateGroundCeil import separate_ceil

def detect_circles(image,dp, min_dist,param1,param2,min_radius,max_radius) -> np.ndarray:
    """
    Detects circles in an image using HoughCircles and draws them filled.
    
    Parameters:
        dp (float): Inverse ratio of the accumulator resolution to the image resolution.
        min_dist (int): Minimum distance between detected centers.
        param1 (int): Higher threshold for the Canny edge detector.
        param2 (int): Accumulator threshold for the circle centers at the detection stage.
        min_radius (int): Minimum circle radius.
        max_radius (int): Maximum circle radius.
    """

    if image is None:
        return None

    # Apply histogram equalization to improve contrast
    image = histeq(image)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using HoughCircles
    circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)

    return circles

if __name__ == "__main__":
    # Load the image
    image_path = "Final Project/dataset/data/easy/rs6.jpg"
    image = cv.imread(image_path)

    if image is None:
        print(f"Error: Image not found at {image_path}")
        exit()

    # Parameters for HoughCircles
    dp = 1
    min_dist = 100    
    param1 = 80  # increase to reduce false positives 
    param2 = 40    # increase to reduce false positives
    min_radius = 10
    max_radius = 65

    # Separate the ceiling from the ground
    image_analyze = image[separate_ceil(image):, :]

    # Detect circles in the image
    circles = detect_circles(image_analyze,dp, min_dist,param1,param2,min_radius,max_radius)

    if circles is not None:
        # print(f"Number of circles detected: {circles.shape[1]}")

        # Convert the circle parameters to integers
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])  # x, y
            center = (center[0], center[1] + separate_ceil(image))
            radius = circle[2]

            print(radius)
            
            # Draw the circle
            cv.rectangle(image, (center[0] - radius, center[1] - radius), (center[0] + radius, center[1] + radius), (0, 255, 0), 3)
            cv.circle(image, center, radius, (0, 0, 255), thickness=2)  # Green filled circle

    # Show the result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
