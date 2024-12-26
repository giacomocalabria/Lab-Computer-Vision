import cv2 as cv
import numpy as np
    
def histeq(image: np.ndarray) -> np.ndarray:
    """
    Equalizes the histogram of the L channel of the LAB color space.

    Args:
        image (numpy.ndarray): image to equalize
    
    Returns:
        numpy.ndarray: equalized image
    """
    # Convert the image to the LAB color space
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)

    # Equalize the histogram of the L channel
    lab_image[:, :, 0] = cv.equalizeHist(lab_image[:, :, 0])

    # Convert the image back to BGR colorspace
    equalized_image = cv.cvtColor(lab_image, cv.COLOR_LAB2BGR)

    return equalized_image