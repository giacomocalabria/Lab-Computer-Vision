import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def detect_circles(image) -> np.ndarray:
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
    
    # Convert the image to unit8 and grayscale for HoughCircles transformation
    uint8_image = (image * 255).astype(np.uint8)

    gray = cv.cvtColor(uint8_image, cv.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv.GaussianBlur(gray, (5, 5), 1)

    # Parameters for HoughCircles
    dp = 1
    min_dist = 15    
    param1 = 90   # increase to reduce false positives 
    param2 = 30   # increase to reduce false positives
    min_radius = 50
    max_radius = 70

    # Detect circles using HoughCircles
    circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)

    if circles is not None:
        print(f"Number of circles detected: {circles.shape[1]}")

    if circles is not None:
        # Convert the circle parameters to integers
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])  # x, y
            radius = circle[2]
            
            # Draw the circle
            cv.circle(image, center, radius, (0, 255, 0), thickness=-1)  # Green filled circle

    # Show the result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return circles


if __name__ == "__main__":

    # Load the image
    image_path = "lab2/data/road8.jpg"
    image = cv.imread(image_path)

    # Detect circles in the image
    detect_circles(image)
