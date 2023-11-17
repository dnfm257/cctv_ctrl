import numpy as np
import supervision as sv
import cv2
from ultralytics import YOLO

# Function to process each frame
def process_frame(frame: np.ndarray, model, zones, zone_annotators, box_annotators) -> np.ndarray:
    results = model(frame, imgsz=1280, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
        mask = zone.trigger(detections=detections)
        detections_filtered = detections[mask]
        frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)
        frame = zone_annotator.annotate(scene=frame)

    return frame

# Load YOLO model
model = YOLO('yolov8s.pt')

# Load video information
VIDEO = "video.mp4"
colors = sv.ColorPalette.default()
video_info = sv.VideoInfo.from_video_path(VIDEO)

# Define polygons and initialize zones
polygons = [
    np.array([
        [718, 595], [927, 592], [851, 1062], [42, 1059]
    ]),
    np.array([
        [987, 595], [1199, 595], [1893, 1056], [1015, 1062]
    ])
]

zones = [
    sv.PolygonZone(
        polygon=polygon,
        frame_resolution_wh=video_info.resolution_wh
    )
    for polygon in polygons
]

# Initialize zone annotators and box annotators
zone_annotators = [
    sv.PolygonZoneAnnotator(
        zone=zone,
        color=colors.by_idx(index),
        thickness=4,
        text_thickness=8,
        text_scale=4
    )
    for index, zone in enumerate(zones)
]

box_annotators = [
    sv.BoxAnnotator(
        color=colors.by_idx(index),
        thickness=4,
        text_thickness=4,
        text_scale=2
    )
    for index in range(len(polygons))
]

# Open video capture
cap = cv2.VideoCapture(VIDEO)

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret or frame is None:
        break

    # Process the frame
    processed_frame = process_frame(frame, model, zones, zone_annotators, box_annotators)

    # Display the processed frame
    cv2.imshow('Processed Frame', processed_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()