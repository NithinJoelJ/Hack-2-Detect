!pip install ultralytics opencv-python numpy torch torchvision torchaudio
!pip install onnxruntime faiss-cpu flask fastapi websockets

!pip install ultralytics
!pip install ultralytics deep-sort-realtime

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from google.colab.patches import cv2_imshow  

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Open video file
cap = cv2.VideoCapture("test.mp4")

# Frame skipping for better speed (set to 1 for no skipping)
frame_skip = 2  
frame_count = 0  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames for performance boost
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    # Run YOLOv8 detection with optimized settings
    results = model.track(frame, conf=0.5, iou=0.5, persist=True)

    detections = []
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coords
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0])  # Class index

            if cls == 0 and conf > 0.5:  # Detect only people
                detections.append([[x1, y1, x2 - x1, y2 - y1], conf, "Person"])

    # Track objects using DeepSORT
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    for track in tracked_objects:
      if not track.is_confirmed():
          continue

      track_id = track.track_id
      x, y, w, h = map(int, track.to_ltwh())

      # Convert track_id to an integer before using it in calculations
      track_id_int = int(track_id)  

      # Assign a unique color per track ID using the integer track_id
      color = (50 * track_id_int % 255, 100 * track_id_int % 255, 200 * track_id_int % 255)

      # Draw bounding box & tracking ID
      cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
      cv2.putText(frame, f"ID {track_id}", (x, y - 10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

      # Show the frame
      cv2_imshow(frame) 
      if cv2.waitKey(1) & 0xFF == ord("q"):
          break

      frame_count += 1  # Increment frame counter

cap.release()
cv2.destroyAllWindows()
