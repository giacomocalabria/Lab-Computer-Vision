import numpy as np
import cv2 as cv
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from matplotlib import pyplot as plt

import xml.etree.ElementTree as ET

def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Get the filename
    filename = root.find("filename").text

    # Extract all objects
    objects = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        bndbox = obj.find("bndbox")
        bbox = {
            "xmin": int(bndbox.find("xmin").text),
            "ymin": int(bndbox.find("ymin").text),
            "xmax": int(bndbox.find("xmax").text),
            "ymax": int(bndbox.find("ymax").text),
        }
        objects.append({"class": class_name, "bbox": bbox})

    return filename, objects

os.environ['OMP_NUM_THREADS'] = '4'

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def extract_features(image):
    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply a gaussian blur to reduce noise
    blurred = cv.GaussianBlur(gray, (9, 9), 1)

    # Use the SIFT feature detector
    sift = cv.SIFT_create()
    _, descriptors = sift.detectAndCompute(blurred, None) # keypoints are not needed

    return descriptors

def build_vocabulary(descriptor_list, k):
    # Stack all descriptors
    all_descriptors = np.vstack(descriptor_list)
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(all_descriptors)
    return kmeans

def compute_histograms(descriptors, kmeans, k):
    # Predict cluster index for each descriptor
    labels = kmeans.predict(descriptors)
    # Create histogram of visual words
    histogram, _ = np.histogram(labels, bins=np.arange(k+1))
    return histogram

def compute_histogram(image, vocabulary, k):
    # Extract features
    features = extract_features(image)
    # Compute histogram
    histogram = compute_histograms(features, vocabulary, k)
    return histogram


def main():
    path_train = "Final Project/dataset/train/images"

    images = load_images_from_folder(path_train)

    print(f"Loaded {len(images)} images")

    # Extract features from all images
    descriptor_list = np.concatenate([extract_features(img) for img in images])

    # Build vocabulary
    k = 100
    vocabulary = build_vocabulary(descriptor_list, k)

    # Compute histograms for all images
    histograms = [compute_histograms(extract_features(img), vocabulary, k) for img in images]

    # Train a SVM classifier
    svm = SVC(kernel='linear', C=1, random_state=42)
    svm.fit(histograms, np.arange(len(images)))

    test = cv.imread("Final Project/circle detection sample/road8.jpg")

    # Compute the histogram for a new image
    histg = compute_histogram(test, vocabulary, k)

    # Predict on new data
    predictions = svm.predict(histg.reshape(1, -1))

    print(predictions)

    plt.imshow(cv.cvtColor(images[9], cv.COLOR_BGR2RGB))
    plt.show()
if __name__ == "__main__":
    main()