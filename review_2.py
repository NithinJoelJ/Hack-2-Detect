import cv2
import numpy as np

# Load YOLO model (weights and configuration files)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize video capture
cap1 = cv2.VideoCapture('sample-1.mp4')  # Video 1 path
cap2 = cv2.VideoCapture('sample-2.mp4')  # Video 2 path

# Check if videos are loaded correctly
if not cap1.isOpened():
    print("Error: Video 1 not found or could not be opened.")
    exit()
if not cap2.isOpened():
    print("Error: Video 2 not found or could not be opened.")
    exit()

print("Videos loaded successfully!")

# Function for detecting objects using YOLO
def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append((x, y, w, h))
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, class_ids, confidences

# Ask the user for the ID of the person to track only once
person_id = input("Enter the ID of the person to track (1 to N): ")
while not person_id.isdigit() or int(person_id) < 1:
    print("Please enter a valid ID (1 to N).")
    person_id = input("Enter the ID of the person to track (1 to N): ")

print(f"Tracking person with ID: {person_id}")

# Initialize tracker
tracker = cv2.TrackerCSRT_create()

# Main loop to read and process video 1
while True:
    ret1, frame1 = cap1.read()
    if not ret1:
        print("End of video 1.")
        break

    # Detect objects in the current frame
    boxes, class_ids, confidences = detect_objects(frame1)

    # Draw bounding boxes around detected people (class_id 0 corresponds to 'person' in coco.names)
    for i in range(len(boxes)):
        if class_ids[i] == 0:  # Person class
            x, y, w, h = boxes[i]
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame1, str(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame with detected people and their IDs
    cv2.imshow("Video 1", frame1)

    # Wait for user to click on a person to start tracking
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Select the bounding box for the entered ID
    selected_bbox = boxes[int(person_id) - 1]  # Get the corresponding bounding box
    tracker.init(frame1, selected_bbox)  # Initialize the tracker for that person

# Tracking the selected person in the second video
while True:
    ret2, frame2 = cap2.read()
    if not ret2:
        print("End of video 2.")
        break

    # Update the tracker with the new frame
    success, bbox = tracker.update(frame2)
    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box for tracked person
    else:
        print("Tracking failed.")

    # Show the frame with tracking information
    cv2.imshow("Video 2", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap1.release()
cap2.release()
cv2.destroyAllWindows()
