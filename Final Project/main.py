import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

from functions.TrafficLightColor import traffic_light_pos_color
from functions.SpeedLimit import get_speed_limit
from functions.HistEq import histeq

# Load the YOLOv8 pre-trained model
model = YOLO("yolo11l.pt")  # Use YOLOv8n, or replace with a traffic sign-specific model

def load_images(folder_path):
    """Load all images from a folder."""
    import os
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    images = [cv2.imread(img_path) for img_path in image_files]
    return images, image_files


def process_image(image, image_name):
    """Process a single image."""
    print(f"Processing: {image_name}")
    #image = histeq(image)  # Apply histogram equalization
    
    results = model.predict(image)
    detections = results[0].boxes.data.cpu().numpy()  # YOLO detection results
    
    for box in detections:
        x1, y1, x2, y2, conf, cls = map(int, box)
        label = results[0].names[cls]
        
        # Traffic light analysis
        if label == "traffic light":
            region = image[y1:y2, x1:x2]
            status = traffic_light_pos_color(region)
            cv2.putText(image, status, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)        
        
        # Speed limit sign analysis
        elif label == "speed limit":
            region = image[y1:y2, x1:x2]
            speed_limit = get_speed_limit(region)
            cv2.putText(image, f"{speed_limit} km/h", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        elif label == "stop sign":
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    
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
        plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Processed: {img_file}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()
