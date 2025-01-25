# Lab-Computer-Vision
Laboratory of Computer Vision - University of Padua. 

These are the labs of the course of Computer Vision at the University of Padua. The labs are written in Python and use the OpenCV library. We have to use the OpenCV library to perform the exercises. The labs are divided into different sections, each one focusing on a different topic of Computer Vision.

Douring the course we have to submit one of the labs and a final project. The final project is a more advanced exercise that requires the use of the concepts learned during the course.

## Lab 0 - Introduction to OpenCV
Introduction to OpenCV and basic operations on images.

## Lab 1.1 - Image Processing

Compute the histograms of the images, perform histogram equalization and histogram matching.

## Lab 1.2 - Image Filtering

Apply different type of filters to the images. We will use the following filters: Gaussian, Median, Bilateral.

## Lab 2 - Hough Transform

Detect lines and circles in the images using the Hough Transform.

# FINAL PROJECT - TBC

This final project focuses on object recognition using deep learning (DL) techniques. The primary goal is to build a system capable of detecting and identifying road sign and traffic light in driving scenarios. 

The project employs both deterministic methods (e.g., Hough Transform) and deep learning models (YOLO) to compare their strengths and weaknesses in various scenarios.

## Features
- Object Detection:
  - YOLO Models: Pre-trained (COCO dataset) and custom-trained models for traffic-related classes.
  - Hough Transform: Detects circular shapes for road signs.
- Object Classification:
  - Speed Limits: Uses OCR (easyocr) to extract speed limit values.
  - Traffic Lights: Grayscale-based method for robust classification of light states.
- Preprocessing: Sky-ground segmentation and noise reduction to enhance detection accuracy.
## Results
- YOLO outperforms the Hough Transform in accuracy and adaptability, especially under complex conditions.
- Brightness-based traffic light classification eliminates issues with color variability.
