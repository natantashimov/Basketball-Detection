import csv
import os

import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class labels
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set the minimum confidence threshold
confidence_threshold = 0.05

# Set the non-maximum suppression threshold
nms_threshold = 0.005

max_distance = 300


def detect_ball_coordinates(frame, last_ball_coordinates):
    # Get the height and width of the frame
    height, width, _ = frame.shape

    # Convert the frame to a blob
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)

    # Set the input blob for the network
    net.setInput(blob)

    # Run forward pass through the network
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Initialize variables
    boxes = []
    confidences = []
    class_ids = []
    ball_coordinates = None

    # Process each output layer
    for output in layer_outputs:
        for detection in output:
            # Get class probabilities
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out non-ball detections and low-confidence detections
            if classes[class_id] == "sports ball" and confidence > confidence_threshold:
                # Scale the bounding box coordinates to the original frame size
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, box_width, box_height) = box.astype("int")

                # Calculate the top-left corner coordinates of the bounding box
                x = int(center_x - (box_width / 2))
                y = int(center_y - (box_height / 2))

                # Add the coordinates, confidence, and class ID to the respective lists
                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Process the remaining boxes after non-maximum suppression
    for i in indices:
        i = i.item(0)
        box = boxes[i]
        x, y, box_width, box_height = box
        center_x = x + (box_width / 2)
        center_y = y + (box_height / 2)

        # Check distance from the last ball detection
        if last_ball_coordinates is not None:
            last_x, last_y = last_ball_coordinates
            distance = np.sqrt((center_x - last_x) ** 2 + (center_y - last_y) ** 2)
            if distance > max_distance:
                continue

        # Ball detection passed filters, update the coordinates
        ball_coordinates = (int(center_x), int(center_y))
        break

    return ball_coordinates


# Input video path
video_path = "C:/Users/natan/PycharmProjects/basketball/sportek/3pt_26.mov"

# Extract video file name without extension
video_file_name = os.path.splitext(os.path.basename(video_path))[0]

# Generate CSV file name with suffix
file_name = f"locations_{video_file_name}_YOLO.csv"

csv_file = open(file_name, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['X', 'Y'])
posList = []

# Output video path
output_path = f"{video_file_name}_YOLO.avi"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

last_ball_coordinates = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    ball_coordinates = detect_ball_coordinates(frame, last_ball_coordinates)

    if ball_coordinates is not None:
        posList.append(ball_coordinates)
        x, y = ball_coordinates
        csv_writer.writerow([1920 - x, 1080 - y])

    for i, pos in enumerate(posList):
        cv2.circle(frame, pos, 3, (0, 255, 0), -1)
        # if i == 0:
        #     cv2.line(frame, pos, pos, (0, 0, 255), 2)
        # else:
        #     cv2.line(frame, pos, posList[i - 1], (0, 0, 255), 2)
        # x, y = ball_coordinates
        # cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)

    out.write(frame)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(len(posList))
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
