# Object-Detection-using-OpenCV


This Python script utilizes the OpenCV library to perform real-time object detection using a webcam. The script uses a pre-trained deep neural network model (SSD MobileNet V3) for detecting objects in the captured frames.

Configuration
The script reads class names from the 'class.names' file. Ensure this file contains the correct class names, one per line.
Model configuration and weights files (ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt and frozen_inference_graph.pb) are specified in the script. Verify their paths.

Troubleshooting
If the script is not detecting objects or displaying incorrect labels, check the class names file, model configuration, and the webcam index.
Ensure the webcam is connected and functioning properly.
