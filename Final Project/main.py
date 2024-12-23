import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import matplotlib.pyplot as plt

# Load the YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")  # Use YOLOv8n, or replace with a traffic sign-specific model
reader = easyocr.Reader(['en'])  # For OCR to extract text from speed signs

def load_images(folder_path):
    """Load all images from a folder."""
    import os
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
    images = [cv2.imread(img_path) for img_path in image_files]
    return images, image_files

def detect_objects(image):
    """Run object detection on the image."""
    results = model.predict(image)
    return results

def analyze_traffic_light(region):
    """Analyze traffic light to determine its status."""
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    mask_yellow = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
    mask_green = cv2.inRange(hsv, (50, 100, 100), (70, 255, 255))
    
    if cv2.countNonZero(mask_red):
        return "Red"
    elif cv2.countNonZero(mask_yellow):
        return "Yellow"
    elif cv2.countNonZero(mask_green):
        return "Green"
    return "Unknown"

def extract_speed_limit(image):
    """Extract the speed limit value from a detected sign."""
    result = reader.readtext(image, detail=0)
    return result[0] if result else "Unknown"

def process_image(image, image_name):
    """Process a single image."""
    print(f"Processing: {image_name}")
    
    results = detect_objects(image)
    detections = results[0].boxes.data.cpu().numpy()  # YOLO detection results
    
    for box in detections:
        x1, y1, x2, y2, conf, cls = map(int, box)
        label = results[0].names[cls]
        
        # Draw bounding box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Traffic light analysis
        if label == "traffic light":
            region = image[y1:y2, x1:x2]
            status = analyze_traffic_light(region)
            cv2.putText(image, status, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Speed limit sign analysis
        elif "speed limit" in label.lower():
            region = image[y1:y2, x1:x2]
            speed_limit = extract_speed_limit(region)
            cv2.putText(image, f"{speed_limit} km/h", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return image

def main():
    # Folder containing "easy" dataset images
    folder_path = "dataset\data\hard"
    
    # Load images
    images, image_files = load_images(folder_path)
    
    for img, img_file in zip(images, image_files):
        processed_image = process_image(img, img_file)
        
        # Show the image
        plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Processed: {img_file}")
        plt.axis('off')
        plt.show()
        
        # Save the processed image
        output_path = img_file.replace("easy_dataset", "output")
        cv2.imwrite(output_path, processed_image)

if __name__ == "__main__":
    main()
