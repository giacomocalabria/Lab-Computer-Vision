# Description: This file is used to detect the speed limit sign in the image provided.

# The function get_speed_limit(ROI) takes a region of interest (ROI) containing a speed limit sign and returns the speed limit detected in the image using OCR and returns the speed limit read from the image.

import numpy as np
import cv2 as cv
import re
import easyocr

import matplotlib.pyplot as plt

def get_speed_limit(ROI) -> int:
    if ROI is None: return "Unknown"

    # convert the ROI to grayscale
    gray = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY)

    # Apply preprocessing to enhance OCR performance
    blurred = cv.GaussianBlur(gray, (5, 5), 1)

    # Use EasyOCR to extract text
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(blurred, allowlist='0123456789')#, text_threshold=1)

    # Extract numbers from the detected text
    for detection in result:
        text = detection[1]  # The detected text
        match = re.search(r'\b\d{1,3}\b', text)  # Search for a number with 1-3 digits
        if match:
            return int(match.group(0))  # Return the first number found

if __name__ == "__main__":
    image = cv.imread("dataset/speed limit sample/100diff.png.jpg")

    speed_limit = get_speed_limit(image)
    print("Speed limit:", speed_limit)