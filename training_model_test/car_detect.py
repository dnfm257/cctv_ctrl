import numpy as np
import supervision as sv
import cv2
from ultralytics import YOLO

# Function to process each frame
def process_frame(frame: np.ndarray, model, zones, zone_annotators, box_annotators) -> np.ndarray:
    # 프레임에 YOLO 객체 감지 적용
    results = model(frame, imgsz=1280, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    #각 영역에 대해 처리
    for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
        #영역을 트리거 후 검출된 것을 필터링
        mask = zone.trigger(detections=detections)
        detections_filtered = detections[mask]
        #바운딩 박스로 프레임 주석처리
        frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)
        #영역으로 프레임 주석처리
        frame = zone_annotator.annotate(scene=frame)

    return frame

# Load YOLO model
model = YOLO('yo/yolov8s.pt')

# Load video information
VIDEO = "video.mp4"
# 주석 색상 초기화
colors = sv.ColorPalette.default()
video_info = sv.VideoInfo.from_video_path(VIDEO)

# 인식영역 정의
polygons = [
    np.array([
        [718, 595], [927, 592], [851, 1062], [42, 1059]
    ]),
    np.array([
        [987, 595], [1199, 595], [1893, 1056], [1015, 1062]
    ])
]
#정의된 영역과 해상도로 초기화
zones = [
    sv.PolygonZone(
        polygon=polygon,
        frame_resolution_wh=video_info.resolution_wh
    )
    for polygon in polygons
]

# 지정된 영역에서 나오는 박스와 텍스트설정
zone_annotators = [
    sv.PolygonZoneAnnotator(
        #영역 지정
        zone=zone,
        #영역색상지정
        color=colors.by_idx(index),
        #선의 두께
        thickness=4,
        #텍스트 두께
        text_thickness=8,
        #텍스트 크기
        text_scale=4
    )
    #zones의 리스트에 따라 다른 속성을 설정함
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
cap = cv2.VideoCapture(0)

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret or frame is None:
        break

    # 프레임 처리
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
