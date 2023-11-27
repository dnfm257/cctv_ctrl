import cv2
import numpy as np
import openvino as ov
import sys
import socket

from twilio.rest import Client
import supervision as sv
from ultralytics import YOLO

import threading
from queue import Queue, Empty

# thread Queue에 대한 접근 제어
input_lock = threading.Lock()
output_lock = threading.Lock()
e = threading.Event()

# 실행 옵션 설정
if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("At least 1 argv needed : [device(CPU or GPU)] [video_path(default webcam 0)]")
    sys.exit(0)

video_path = 0

if len(sys.argv) == 3:
    video_path = sys.argv[2]

select_device = sys.argv[1]

frame_size = np.random.random((480, 640, 3))
key = None

# BoxAnnotator 클래스 정의
class BoxAnnotator(sv.BoxAnnotator):
    def __init__(self, color, thickness=2, text_thickness=1, text_scale=1):
        self.color = ensure_color_format(color)
        self.thickness = thickness
        self.text_thickness = text_thickness
        self.text_scale = text_scale

    def annotate_box_only(self, scene, detection):
        x1, y1, x2, y2 = map(int, detection[:4])
        cv2.rectangle(scene, (x1, y1), (x2, y2), self.color, self.thickness)
        return scene

# Load the OpenVINO model
def load_model():
    model_xml_path = "model_yolox_v3/openvino/openvino.xml"
    model_bin_path = "model_yolox_v3/openvino/openvino.bin"
    
    core = ov.Core()
    model = core.read_model(model=model_xml_path, weights=model_bin_path)
    
    # static shape convert
    if select_device == 'GPU':
        model.reshape([1, 3, 416, 416]) 
        
    compiled_model = core.compile_model(model=model, device_name=select_device)

    return compiled_model

# webcam or video input
def init_camera(buf_size=3):
    global video_path, frame_size
        
    cap = cv2.VideoCapture(video_path)
    #cap.set(cv2.CAP_PROP_BUFFERSIZE, buf_size)  # 버퍼 크기 설정
    #cap.set(cv2.CAP_PROP_POS_MSEC, timeout) # (ms) timeout
    
    # Get webcam resolution
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = np.random.random((frame_height, frame_width, 3))

    if not cap.isOpened():
        print("Error with Camera")
        
    return cap

# Set the dimensions explicitly
def get_shape(compiled_model):
    input_layer = compiled_model.input(0)
    
    # dynamic shape
    if select_device == 'CPU':
        partial_shape = input_layer.get_partial_shape()
        _, _, H, W = partial_shape
        h, w = H.get_length(), W.get_length()
    # static shape
    else:
        _, _, h, w = input_layer.shape
    
    return h, w

# Preprocess the frame to match the input requirements of the model
def preprocess(frame, W, H):
    resized_frame = cv2.resize(frame, (W, H))
    # 높이, 너비, 색상 -> 색상, 높이, 너비 변경 후 맨 앞에 1추가 => 모델 input shape와 동일하게 만들어줌
    input_frame = np.expand_dims(resized_frame.transpose(2, 0, 1), axis=0)

    return resized_frame, input_frame

# 원본 frame 대비 resize된 frame 비율 측정
def adjust_ratio(frame, resized_frame):
    (real_y, real_x), (resized_y, resized_x) = frame.shape[:2], resized_frame.shape[:2]
    ratio_x, ratio_y = (real_x / resized_x), (real_y / resized_y)
    
    return ratio_x, ratio_y

# 인식박스 만들기
def creat_boxes(frame, x_min, y_min, x_max, y_max, text, color, rect=True):
    if rect:
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
    cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# 차량 인식영역 정의
def define_polygons(ratio_x, ratio_y):
    # road1 polygon1
    polygons_road1 = np.array([
        [int(725 / ratio_x), int(210 / ratio_y)],
        [int(852 / ratio_x), int(210 / ratio_y)],
        [int(676 / ratio_x), int(943 / ratio_y)],
        [int(52 / ratio_x), int(856 / ratio_y)],
        [int(721 / ratio_x), int(213 / ratio_y)]
    ], dtype=np.int32)

    # ROAD2의 polygon1 수정
    polygons_road2 = np.array([
        [int(742 / ratio_x), int(416 / ratio_y)],
        [int(1052 / ratio_x), int(412 / ratio_y)],
        [int(1352 / ratio_x), int(930 / ratio_y)],
        [int(387 / ratio_x), int(941 / ratio_y)],
        [int(740 / ratio_x), int(418 / ratio_y)]
    ], dtype=np.int32)

    # ROAD3의 polygon1 수정
    polygons_road3 = np.array([
        [int(841 / ratio_x), int(338 / ratio_y)],
        [int(1168 / ratio_x), int(335 / ratio_y)],
        [int(1759 / ratio_x), int(892 / ratio_y)],
        [int(776 / ratio_x), int(905 / ratio_y)],
        [int(842 / ratio_x), int(347 / ratio_y)]
    ], dtype=np.int32)

    # road4의 새로운 polygon2 수정
    polygons_road4 = np.array([
        [int(895 / ratio_x), int(190 / ratio_y)],
        [int(983 / ratio_x), int(178 / ratio_y)],
        [int(1678 / ratio_x), int(607 / ratio_y)],
        [int(999 / ratio_x), int(671 / ratio_y)],
        [int(897 / ratio_x), int(187 / ratio_y)]
    ], dtype=np.int32)
    
    polygon_options = [[polygons_road1], [polygons_road2], [polygons_road3], [polygons_road4]]

    return polygon_options

# 인식영역, 인식박스 init
def init_annotators(current_polygons, colors):
    global frame_size
    # Convert current polygons to PolygonZone objects
    current_zones = [
        sv.PolygonZone(polygon=polygon, frame_resolution_wh=(frame_size.shape[1], frame_size.shape[0]))
        for polygon in current_polygons
    ]

    # Initialize zone annotators and box annotators for both polygons
    zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone,
            color=ensure_color_format(colors.by_idx(index % 8)),
            thickness=4,
            text_thickness=8,
            text_scale=4
        )
        for index, zone in enumerate(current_zones)
    ]

    # box_annotators 리스트 초기화
    box_annotators = [
        BoxAnnotator(
            color=ensure_color_format(colors.by_idx(index % 8)),
            thickness=4,
            text_thickness=4,
            text_scale=2
        )
        for index, _ in enumerate(current_polygons)
    ]
    
    return current_zones, zone_annotators, box_annotators

# 색상 형식 확인 및 변환 함수
def ensure_color_format(color):
    if isinstance(color, tuple) and len(color) == 3 and all(isinstance(c, int) for c in color):
        return color
    else:
        return (255, 0, 0)  # 기본 색상: 빨간색
    
# 폴리곤에 숫자 지우는 함수 정의
def draw_polygon_without_count(scene, polygon, zone_annotator, thickness):
    cv2.polylines(scene, [polygon], isClosed=True, color=zone_annotator.color, thickness=thickness)
    return scene

# 옵션 조정
def set_options(frame, polygon_options, colors, polygons):
    global key
    
    # default values
    current_polygons = polygons
    show_polygons = True
    show_boxes = True
    update_annotators = True
    
    if key == ord('1'):
        current_polygons = polygon_options[0]
        show_polygons = True
        show_boxes = True
        update_annotators = True
    elif key == ord('2'):
        current_polygons = polygon_options[1]
        show_polygons = True
        show_boxes = True
        update_annotators = True
    elif key == ord('3'):
        current_polygons = polygon_options[2]
        show_polygons = True
        show_boxes = True
        update_annotators = True
    elif key == ord('4'):
        current_polygons = polygon_options[3]
        show_polygons = True
        show_boxes = True
        update_annotators = True
    elif key == ord('5'):
        current_polygons = [] # 폴리곤을 숨깁니다.
        show_polygons = False
        show_boxes = False
        update_annotators = True
    else:
        update_annotators = False
        
    if update_annotators:
        # Update current zones and annotators
        current_zones = [
            sv.PolygonZone(polygon=polygon, frame_resolution_wh=(frame.shape[1], frame.shape[0]))
            for polygon in current_polygons
        ]
        zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=zone,
                color=ensure_color_format(colors.by_idx(index % 8)),
                thickness=4,
                text_thickness=8,
                text_scale=4
            )
            for index, zone in enumerate(current_zones)
        ]
        box_annotators = [
            BoxAnnotator(
                color=ensure_color_format(colors.by_idx(index % 8)),
                thickness=4,
                text_thickness=4,
                text_scale=2
            )
            for index, _ in enumerate(current_polygons)
        ]
        update_annotators = False
        
        return (current_polygons, show_boxes, show_polygons, current_zones, zone_annotators, box_annotators)
    
    return (current_polygons, show_boxes, show_polygons)

def create_socket():
    global server_socket
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('10.10.141.24', 5001))
    server_socket.listen(5)

def connect_client(server_socket):
    global client_socket
    
    client_socket, addr = server_socket.accept()

# 차량 통행량 발신
def send_msg(client_socket, msg):
    global server_socket

    data = (f"{msg}\r\n")
    #print(data)
    client_socket.send(data.encode('utf-8'))

# 모바일 전송 코드
def send_sms(flags):
    if flags == 1:
        data = "**사거리 사고발생 경찰력 총동원"
    elif flags == 2:
        data = "**사거리 사고발생 구급차 출동요망"
    else:
        return
    
    account_sid = 'AC830601052a526b757f23cac741e8becb'
    auth_token = ''
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        from_='+12563685788',
        body=data.encode(),
        to='+821031198106'
    )

# 사고 detect
def detect_accident(input_queue, output_queue):    
    # 레이블 정의
    labels = {
        "0": {"name": "fall", "color": (255, 0, 0)}, # 넘어진 사람
        "1": {"name": "moderate", "color": (0, 255, 0)}, # 경미한 사고
        "2": {"name": "severe", "color": (0, 0, 255)} # 중대한 사고
    }
    compiled_model = load_model()
    
    # Define the output layers
    output_layer_labels = compiled_model.output("labels")
    output_layer_boxes = compiled_model.output("boxes")
    
    H, W = get_shape(compiled_model)
    
    flag = True
    
    while True:
        with input_lock:
            # cam frame input
            frame = input_queue.get()
        
        # 종료
        if frame is None:
            break
        
        e.set()
        
        resized_frame, input_frame = preprocess(frame, W, H)
    
        # 화면비율을 통한 좌표수정
        ratio_x, ratio_y = adjust_ratio(frame, resized_frame)
        
        # Perform inference
        results = compiled_model([input_frame])[output_layer_boxes] # 좌표
        class_ids = compiled_model([input_frame])[output_layer_labels] # class id
        
        valid_detections = np.where(results[0][:, 4] > 0.8)[0]

        for i in valid_detections:
            x_min, y_min, x_max, y_max, score = results[0][i] # 좌표, 신뢰도 추출
            class_id = str(class_ids[0][i])  # class id 추출

            #좌상단, 우하단 좌표
            x_min = int(max(x_min * ratio_x, 10))
            y_min = int(y_min * ratio_y)
            x_max = int(x_max * ratio_x)
            y_max = int(y_max * ratio_y)

            # 인식한 객체에 대한 레이블 정보 가져오기
            label_name = labels[class_id]["name"]
            color = labels[class_id]["color"]
            
            if flag:
                if label_name == "moderate" or label_name == "severe":
                    send_sms(1)
                elif label_name == "fall":
                    send_sms(2)
                
                flag = False
            
            st = f"{label_name}  {score:.2f}"
            print(st)
            creat_boxes(frame, x_min, y_min, x_max, y_max, st, color)
        
        with output_lock:
            output_queue.put(('Webcam Object Detection', frame))
        
# 차량 트래픽 감지
def detect_traffic(input_queue, output_queue):
    global frame_size, client_socket
    
     # Load YOLO model
    model = YOLO('yolov8s.pt')
    
    # Load colors
    colors = sv.ColorPalette.default()
    
    # Original image resolution
    original = np.random.random((1006 ,1796, 3))
    
    # 영상에 사용될 polygons(roi) 정의
    ratio_x, ratio_y = adjust_ratio(original, frame_size)
    polygon_options = define_polygons(ratio_x, ratio_y)
    current_polygons = polygon_options[0] # default: polygons_road1
    
    current_zones, zone_annotators, box_annotators = init_annotators(current_polygons, colors)
    
    # 인식객체수 표시 여부를 결정하는 플래그 추가
    show_count=False
    object_count = 0

    # Car class ID for filtering (assuming 'car' class ID is 2)
    car_class_id = 2
    
    while True:
        # 선행 스레드가 끝날 때까지 대기
        e.wait()
        
        with input_lock:
            frame = input_queue.get()
            
        e.clear()
        
        # 폴리곤(roi)여부, 폴리곤(roi)모양, 인식박스 여부,
        options = set_options(frame, polygon_options, colors, current_polygons)
        if len(options) == 3:
            current_polygons, show_boxes, show_polygons = options
        else:
            current_polygons, show_boxes, show_polygons, current_zones, zone_annotators, box_annotators = options
        
        results = model(frame, imgsz=640, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        for zone, zone_annotator, box_annotator in zip(current_zones, zone_annotators, box_annotators):
            # Car detections 필터링
            car_detections = [d for d, class_id in zip(detections.xyxy, detections.class_id) if class_id == car_class_id]

            # 각 탐지에 대해 mask 계산
            mask = np.array([zone.trigger(detections=sv.Detections(
                xyxy=np.array([d]), 
                confidence=np.array([detections.confidence[idx]]), 
                class_id=np.array([class_id])
                )) for idx, (d, class_id) in enumerate(zip(detections.xyxy, detections.class_id)) if class_id == car_class_id])
            detections_filtered = [d for d, m in zip(car_detections, mask) if m]
            object_count = len(detections_filtered)  # 각 zone에 있는 차량 수 계산
            
            # process_frame 함수 내
            if show_boxes:
                for detection in detections_filtered:
                    frame = box_annotator.annotate_box_only(scene=frame, detection=detection)

            # 폴리곤 조건부 주석 처리
            if show_polygons:
                if show_count:
                    frame = zone_annotator.annotate(scene=frame)
                else:
                    frame = draw_polygon_without_count(
                        scene=frame,
                        polygon=zone.polygon,
                        zone_annotator=zone_annotator,
                        thickness=zone_annotator.thickness
                    )
            
            text = f"Car Count: {object_count}"
            color = (255, 0, 255)
            creat_boxes(frame, int(zone.polygon[0][0]-50), int(zone.polygon[0][1] + 50), 0, 0, text, color, rect=False)
        
        with output_lock:
            output_queue.put(("Webcam traffic Detection", frame))
            
        e.clear()
        send_msg(client_socket, object_count)

def main():
    global key, server_socket, client_socket
    fps = 30 # 영상 FPS 조절
    
    input_queue = Queue(maxsize=6) # 영상 input queue
    output_queue = Queue() # 후처리 영상 output queue
    
    detection_thread = threading.Thread(
        target=detect_accident, 
        args=(input_queue, output_queue))
    traffic_thread = threading.Thread(
        target=detect_traffic,
        args=(input_queue, output_queue)
    )
    
    detection_thread.start()
    traffic_thread.start()
    
    create_socket()
    connect_client(server_socket)
    cap = init_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        key = cv2.waitKey(1000 // fps) & 0xFF
        if key == ord('q'):
            break
        
        # main함수와 스레드간 속도 동기화를 위해 블로킹처리
        input_queue.put(frame, block=True)
        
        # get frame from queue
        try:
            with output_lock:
                data = output_queue.get_nowait()
        except Empty:
            continue
            
        name, det_frame = data
        
        cv2.imshow(name, det_frame)
        
        output_queue.task_done()
        
    input_queue.put(None)
    
    server_socket.close()
    client_socket.close()
    
    detection_thread.join()
    traffic_thread.join()
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()