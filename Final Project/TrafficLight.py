import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from functions.TrafficLightColor import traffic_light_pos_color, traffic_light_color

stop_light = cv.CascadeClassifier('CascadeClassifier/TrafficLight_HAAR_16Stages.xml')

def load_images(folder_path):
    """Load all images from a folder."""
    import os
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    images = [cv.imread(img_path) for img_path in image_files]
    return images, image_files

def process_image(image, image_name):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    stop_lights = stop_light.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in stop_lights:
        roi = image[y:y+h, x:x+w]
        color = traffic_light_color(roi)
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv.putText(image, color, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
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