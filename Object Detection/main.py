import cv2

# Set the threshold to detect objects
thres = 0.45

# Open the webcam (camera index 1)
cap = cv2.VideoCapture(1) # use 0 in index to use the webcam
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height
cap.set(10, 70)   # Set brightness

# Read class names from the 'class.names' file
classNames = []
classFile = 'class.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n')

# Set paths for model configuration and weights
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Load the pre-trained model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Main loop for object detection
while True:
    # Read a frame from the webcam
    success, img = cap.read()

    # Check if the frame is empty
    if not success or img is None or img.size == 0:
        print("Failed to capture frame. Exiting...")
        break

    # Perform object detection on the frame
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    # Print class IDs and bounding boxes
    print(classIds, bbox)

    # If objects are detected, draw bounding boxes and labels
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Draw bounding box
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

            # Display class label
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            print("Detected Class Indices:", classIds)
            #print("Class Names:", classNames)

            # Display confidence score
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow('output', img)

    # Wait for a key press (1ms delay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
