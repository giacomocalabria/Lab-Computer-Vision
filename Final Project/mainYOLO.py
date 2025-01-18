import cv2 as cv
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

from functions.TrafficLightColor import traffic_light_pos_color
from functions.SpeedLimit import get_speed_limit
from functions.CircleHough import detect_circles
from functions.SeparateGroundCeil import separate_ceil
from functions.HistEq import histeq

# Load the YOLOv8 pre-trained model
model = YOLO("yolo11x.pt")  # Use YOLOv8n, or replace with a traffic sign-specific model

def load_images(folder_path):
    """Load all images from a folder."""
    import os
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    images = [cv.imread(img_path) for img_path in image_files]
    return images, image_files


# Parameters for HoughCircles
dp = 1
min_dist = 100    
param1 = 80     # increase to reduce false positives 
param2 = 40     # increase to reduce false positives
min_radius = 10
max_radius = 65

def process_image(image, image_name):
    """Process a single image."""
    print(f"Processing: {image_name}")
    #image = histeq(image)  # Apply histogram equalization
    
    # Detect object in the image with YOLO
    print(image.shape)
    resultsYOLO = model.predict(image,imgsz=image.shape[:2], classes=[9,11])

    # Detect circles in the image with HoughCircles
    circles = detect_circles(image[separate_ceil(image):, :],dp, min_dist,param1,param2,min_radius,max_radius)

    cv.line(image, (0, separate_ceil(image)), (image.shape[1], separate_ceil(image)), (0, 0, 255), 2)

    detections = resultsYOLO[0].boxes.data.cpu().numpy()  # YOLO detection results
    for box in detections:
        x1, y1, x2, y2, conf, cls = map(int, box)
        label = resultsYOLO[0].names[cls]
        
        # Traffic light analysis
        if label == "traffic light":
            region = image[y1:y2, x1:x2]
            status = traffic_light_pos_color(region)
            cv.putText(image, status, (x1, y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(image, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        elif label == "stop sign":
            cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(image, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if circles is not None: # HoughCircles detected circles
        # print(f"Number of circles detected: {circles.shape[1]}")

        # Convert the circle parameters to integers
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])  # x, y
            center = (center[0], center[1] + separate_ceil(image))
            radius = circle[2]

            # Extract the ROI for speed limit sign
            region = image[center[1] - radius -15: center[1] + radius +15, center[0] - radius -15: center[0] + radius+15]
            speed_limit = get_speed_limit(region)

            if speed_limit is not None:
                cv.putText(image, f"{speed_limit} km/h", (center[0] - radius, center[1] + radius + 40), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                cv.rectangle(image, (center[0] - radius - 5, center[1] - radius - 5), (center[0] + radius +5, center[1] + radius+5), (255, 255, 0), 3)
            #else:
            #    cv.putText(image, "Generic road sign", (center[0] - radius, center[1] + radius + 40), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        
    return image

def main():
    # Folder containing "easy" test image dataset
    folder_path = "dataset/data/easy"

    # Folder containing "hard" test image dataset
    # folder_path = "dataset/data/hard"
    
    # Load images
    images, image_files = load_images(folder_path)
    
    for img, img_file in zip(images, image_files):
        processed_image = process_image(img, img_file)
        
        # Show the image
        plt.imshow(cv.cvtColor(processed_image, cv.COLOR_BGR2RGB))
        plt.title(f"Processed: {img_file}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()
