import cv2
import numpy as np
import openvino as ov

import threading
from queue import Queue, Empty

def load_model(device='CPU'):
    # Load the OpenVINO model
    model_xml_path = "model_yolox_v2_2/openvino/openvino.xml"
    model_bin_path = "model_yolox_v2_2/openvino/openvino.bin"
    
    core = ov.Core()
    model = core.read_model(model=model_xml_path, weights=model_bin_path)
    compiled_model = core.compile_model(model=model, device_name=device)

    return compiled_model

def init_camera(camera_index=0, buf_size=5, timeout=1000):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buf_size)  # 버퍼 크기 설정
    cap.set(cv2.CAP_PROP_POS_MSEC, timeout) # (ms) timeout
    if not cap.isOpened():
        print("Error with Camera")
        
    return cap

def get_shape(compiled_model):
    input_layer = compiled_model.input(0)
    
    # Set the dimensions explicitly
    partial_shape = input_layer.get_partial_shape()
    _, _, H, W = partial_shape
    h, w = H.get_length(), W.get_length()
    
    return h, w

# Preprocess the frame to match the input requirements of the model
def preprocess(frame, W, H):
    resized_frame = cv2.resize(frame, (W, H))
    input_frame = np.expand_dims(resized_frame.transpose(2, 0, 1), axis=0)

    return resized_frame, input_frame

def adjust_ratio(frame, resized_frame):
    (real_y, real_x), (resized_y, resized_x) = frame.shape[:2], resized_frame.shape[:2]
    ratio_x, ratio_y = (real_x / resized_x), (real_y / resized_y)
    
    return ratio_x, ratio_y

def creat_boxes(frame, x_min, y_min, x_max, y_max, text, color):
    # 박스 만들기
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

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
    
    while True:
        # frame input
        frame = input_queue.get()
        
        # 종료
        if frame is None:
            break
    
        resized_frame, input_frame = preprocess(frame, W, H)
    
        # 화면비율을 통한 좌표수정
        ratio_x, ratio_y = adjust_ratio(frame, resized_frame)
    
        # Perform inference
        results = compiled_model([input_frame])[output_layer_boxes] # 좌표
        class_ids = compiled_model([input_frame])[output_layer_labels] # class id
    
        for i, detection in enumerate(results[0]):
    
            if np.all(detection == 0):
                continue
        
            x_min, y_min, x_max, y_max, score = detection # 좌표, 신뢰도 추출
            class_id = str(class_ids[0][i])  # class id 추출

            if (score > 0.8) and (class_id in labels): # 신뢰도 및 라벨
                #좌상단, 우하단 좌표
                x_min = int(max(x_min * ratio_x, 10))
                y_min = int(y_min * ratio_y)
                x_max = int(x_max * ratio_x)
                y_max = int(y_max * ratio_y)

                # 해당 레이블에 대한 정보 가져오기
                label_name = labels[class_id]["name"]
                color = labels[class_id]["color"]

                st = f"{label_name}  {score:.2f}"
                print(st)
                creat_boxes(frame, x_min, y_min, x_max, y_max, st, color)
            
        output_queue.put(('Webcam Object Detection', frame))


def main():
    input_queue = Queue()
    output_queue = Queue()
    
    detection_thread = threading.Thread(target=detect_accident, 
                                        args=(input_queue, output_queue))
    detection_thread.start()
    
    cap = init_camera()

    while True:
        ret, frame = cap.read()
        if not ret or (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        
        input_queue.put(frame)
        
        # get frame from queue
        try:
            data = output_queue.get_nowait()
        except Empty:
            continue
            
        name, det_frame = data

        cv2.imshow(name, det_frame)
        
        output_queue.task_done()
        
    input_queue.put(None)
    detection_thread.join()
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()