import cv2
import numpy as np
import openvino as ov

# Load the OpenVINO model
model_xml_path = "model/openvino.xml"
model_bin_path = "model/openvino.bin"

core = ov.Core()
model = core.read_model(model=model_xml_path, weights=model_bin_path)
compiled_model = core.compile_model(model=model, device_name='CPU')

# Define the input layer
input_layer = compiled_model.input(0)
output_layer_labels = compiled_model.output("labels")
output_layer_boxes = compiled_model.output("boxes")
#_, C, H, W = 1, 3, 416, 416  # Set the dimensions explicitly
partial_shape = input_layer.get_partial_shape()
_, _, H, W = partial_shape

# 레이블 정의
labels = {
    "0": {"name": "fall", "color": (255, 0, 0)}, # 넘어진 사람
    "1": {"name": "moderate", "color": (0, 255, 0)}, # 경미한 사고
    "2": {"name": "severe", "color": (0, 0, 255)} # 중대한 사고
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame to match the input requirements of the model
    resized_frame = cv2.resize(frame, (W.get_length(), H.get_length()))
    input_frame = np.expand_dims(resized_frame.transpose(2, 0, 1), axis=0)

    # Perform inference
    results = compiled_model([input_frame])[output_layer_boxes] # 좌표
    class_ids = compiled_model([input_frame])[output_layer_labels] # class id
    
    # 화면비율을 통한 좌표수정
    (real_y, real_x), (resized_y, resized_x) = frame.shape[:2], resized_frame.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
    
    # Process the results
    for i, detection in enumerate(results[0]):
    
        if np.all(detection == 0):
            continue

        x_min, y_min, x_max, y_max, score = detection
        class_id = str(class_ids[0][i])  # class id 추출

        if score > 0.8 and class_id in labels:  # 신뢰도 및 라벨
            x_min = int(max(x_min * ratio_x, 10))
            y_min = int(y_min * ratio_y)
            x_max = int(x_max * ratio_x)
            y_max = int(y_max * ratio_y)

            # 해당 레이블에 대한 정보 가져오기
            label_name = labels[class_id]["name"]
            color = labels[class_id]["color"]

            st = f"{label_name}  {score:.2f}"
            # 박스 만들기
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, st, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Webcam Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()