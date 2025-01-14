import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from SpeedLimit import get_speed_limit
from CircleHough import detect_circles
from SeparateGroundCeil import separate_ceil

def load_images(folder_path):
    """Load all images from a folder."""
    import os
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    images = [cv.imread(img_path) for img_path in image_files]
    return images, image_files

def process_image(image, image_name):
    # Parameters for HoughCircles
    dp = 1
    min_dist = 100    
    param1 = 80     # increase to reduce false positives 
    param2 = 40     # increase to reduce false positives
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

            # Extract the ROI for speed limit sign
            region = image[center[1] - radius -10: center[1] + radius +10, center[0] - radius -10: center[0] + radius+10]
            speed_limit = get_speed_limit(region)

            cv.rectangle(image, (center[0] - radius, center[1] - radius), (center[0] + radius, center[1] + radius), (0, 255, 0), 3)

            if speed_limit is not None:
                cv.putText(image, f"{speed_limit} km/h", (center[0] - radius, center[1] + radius + 40), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    return image


def main():
    folder_path = "dataset/data/easy"

    # Load images
    images, image_files = load_images(folder_path)

    print(f"Number of images loaded: {len(images)}")
    
    for img, img_file in zip(images, image_files):
        processed_image = process_image(img, img_file)
        
        # Show the image
        plt.imshow(cv.cvtColor(processed_image, cv.COLOR_BGR2RGB))
        plt.title(f"Processed: {img_file}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()