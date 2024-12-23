# This file is used to determine the traffic light status based on the color of the light. The function analyze_traffic_light(region) takes a region of interest (ROI) containing a traffic light and returns the status of the light (Red, Yellow, Green, or Unknown). 

# The function uses OpenCV to convert the ROI to the HSV color space and then applies color masks to detect red, yellow, and green colors. The function returns the status of the traffic light based on the color detected.

import cv2 as cv
import numpy as np

# define the color ranges for red, yellow, and green
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
lower_yellow = np.array([20, 50, 50])
upper_yellow = np.array([30, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([90, 255, 255])

def traffic_light_color(ROI):
    if ROI is None:
        return "Unknown" # no image provided

    # convert the ROI to HSV color space
    hsv = cv.cvtColor(ROI, cv.COLOR_BGR2HSV)

    # create masks for red, yellow, and green colors
    mask_red = cv.inRange(hsv, lower_red, upper_red)
    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv.inRange(hsv, lower_green, upper_green)

    # count the number of non-zero pixels in each mask
    red_count = cv.countNonZero(mask_red)
    yellow_count = cv.countNonZero(mask_yellow)
    green_count = cv.countNonZero(mask_green)

    color_counts = {
        "Red": red_count,
        "Yellow": yellow_count,
        "Green": green_count
    }

    sorted_colors = sorted(color_counts.items(),  key=lambda x: x[1], reverse=True)

    print("Sorted colors:", sorted_colors)

    prominent_color, max_count = sorted_colors[0]

    total_pixel = ROI.shape[0] * ROI.shape[1]
    threshold = 0.001 * total_pixel

    # Check for red + yellow combination (British standard)
    if prominent_color == "Red" and max_count > threshold and yellow_count > threshold:
        return "Yellow with Red"
    
    if max_count > threshold:
        return prominent_color
    else:
        return "Unknown"
    
def traffic_light_pos_color(ROI):
    if ROI is None:
        return "Unknown" # no image provided

    # convert the ROI to HSV color space
    ROI = cv.cvtColor(ROI, cv.COLOR_BGR2HSV)

    # split the image into three regions
    height, width = ROI.shape[:2]
    isVertical = height > width # check if the image is vertical or horizontal

    if isVertical:
        red_region = ROI[0:int(height/3), :]                    # top
        yellow_region = ROI[int(height/3):int(2*height/3), :]   # middle
        green_region = ROI[int(2*height/3):, :]                 # bottom
    
    else:
        red_region = ROI[:, 0:int(width/3)]                     # left
        yellow_region = ROI[:, int(width/3):int(2*width/3)]     # middle
        green_region = ROI[:, int(2*width/3):]                  # right
    
    # create masks for red, yellow, and green colors
    mask_red = cv.inRange(red_region, lower_red, upper_red)
    mask_yellow = cv.inRange(yellow_region, lower_yellow, upper_yellow)
    mask_green = cv.inRange(green_region, lower_green, upper_green)

    # count the number of non-zero pixels in each mask
    red_count = cv.countNonZero(mask_red)
    yellow_count = cv.countNonZero(mask_yellow)
    green_count = cv.countNonZero(mask_green)
    
    color_counts = {
        "Red": red_count,
        "Yellow": yellow_count,
        "Green": green_count
    }

    sorted_colors = sorted(color_counts.items(),  key=lambda x: x[1], reverse=True)
    prominent_color, max_count = sorted_colors[0]

    threshold = int(np.mean([red_count, yellow_count, green_count]))

    # Check for red + yellow combination (British standard)
    if prominent_color == "Red" and max_count > threshold and yellow_count > threshold:
        return "Yellow with Red"
    
    if max_count > threshold:
        return prominent_color
    else:
        return "Unknown"

if __name__ == "__main__":
    # Load an image containing a traffic light
    ROI = cv.imread("redyellow.png")

    # Analyze the traffic light status
    status = traffic_light_pos_color(ROI)
    print("Traffic light status:", status)
