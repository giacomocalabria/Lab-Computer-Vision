# Description: This file is used to detect the speed limit sign in the image provided.

# The function get_speed_limit(ROI) takes a region of interest (ROI) containing a speed limit sign and returns the speed limit detected in the image using OCR and returns the speed limit read from the image.

import numpy as np
import cv2 as cv
import easyocr

def get_speed_limit(ROI):
    if ROI is None:
        return "Unknown" # no image provided

    # convert the ROI to grayscale
    gray = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY)

    # Apply preprocessing to enhance OCR performance
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Use EasyOCR to extract text
    reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a GPU
    result = reader.readtext(binary)

    # Extract numbers from the detected text
    for detection in result:
        text = detection[1]  # The detected text
        if text.isdigit():
            return int(text)

if __name__ == "__main__":
    image = cv.imread("speed limit sample/100.png")

    speed_limit = get_speed_limit(image)
    print("Speed limit:", speed_limit)