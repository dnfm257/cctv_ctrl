import cv2
import numpy as np
from openvino.runtime import Core

# Load the OpenVINO model
model_xml_path = "modeling/model/openvino.xml"
model_bin_path = "modeling/model/openvino.bin"

core = Core()
model = core.read_model(model=model_xml_path, weights=model_bin_path)
compiled_model = core.compile_model(model=model, device_name='CPU')

# Define the input layer
input_layer = compiled_model.input(0)
_, C, H, W = 1, 3, 416, 416  # Set the dimensions explicitly

# Define the label information
labels = {
    "1": {"name": "fall", "color": (229, 167, 236)},
    "2": {"name": "moderate", "color": (6, 76, 236)},
    "3": {"name": "severe", "color": (174, 165, 116)}
}

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame to match the input requirements of the model
    resized_frame = cv2.resize(frame, (W, H))
    input_frame = np.expand_dims(resized_frame.transpose(2, 0, 1), axis=0)

    # Perform inference
    results = compiled_model([input_frame])[compiled_model.output("boxes")]
    class_ids = compiled_model([input_frame])[compiled_model.output("labels")]

    # Process the results
    for i, detection in enumerate(results[0]):
    # Skip the padding detection with all zeros
        if np.all(detection == 0):
            continue

        x_min, y_min, x_max, y_max, score = detection
        class_id = class_ids[0][i]  # Get the corresponding class ID for each detection

        if score > 0.3 and str(class_id) in labels:  # Adjust the confidence threshold as needed.
        # Convert coordinates to integers
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        # Look up label name and color
            label_name = labels[str(class_id)]["name"]
            color = labels[str(class_id)]["color"]

        # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, label_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame
    cv2.imshow('Webcam Object Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
