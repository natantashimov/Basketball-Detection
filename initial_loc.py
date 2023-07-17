import cv2
import numpy as np
import csv

# Path to your video file
video_path = './court/splash.mp4'

# Define the size and circularity thresholds for filtering
min_ball_radius = 13
max_ball_radius = 18
circularity_threshold = 0.8

# Define the maximum distance between consecutive ball detections
max_distance = 100

# Define the range for initial ball detection location
initial_detection_range_x = (900, 1200)  # Adjust the values as needed
initial_detection_range_y = (350, 500)  # Adjust the values as needed
# initial_detection_range_x = (450, 600)  # Adjust the values as needed
# initial_detection_range_y = (250, 500)  # Adjust the values as needed

def detect_ball_coordinates(frame, last_ball_coordinates):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Hough Circle Transform to detect circles (balls)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1, minDist=50, param1=50, param2=25,
        minRadius=min_ball_radius, maxRadius=max_ball_radius
    )
    if circles is not None:
        # Extract the coordinates and radius of the detected balls
        circles = np.round(circles[0, :]).astype("int")
        print(circles)

        # Filter and verify the ball detections
        for (x, y, radius) in circles:
            # Check size
            if radius < min_ball_radius or radius > max_ball_radius:
                continue

            # Check initial detection range
            if last_ball_coordinates is None:
                if not (initial_detection_range_x[0] <= x <= initial_detection_range_x[1] and
                        initial_detection_range_y[0] <= y <= initial_detection_range_y[1]):
                    continue

            # Create a mask for the current circle
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), radius, 255, -1)

            # Calculate circularity
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Ensure at least one contour was found
            if len(contours) == 0:
                continue

            perimeter = cv2.arcLength(contours[0], True)
            area = cv2.contourArea(contours[0])
            circularity = 4 * np.pi * area / (perimeter ** 2)

            # Check circularity
            if circularity < circularity_threshold:
                continue

            # Check distance from the last ball detection
            if last_ball_coordinates is not None:
                last_x, last_y = last_ball_coordinates
                distance = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)
                if distance > max_distance:
                    continue

            # Ball detection passed filters, return the coordinates
            return (x, y)

    return None


# Load the video file
cap = cv2.VideoCapture(video_path)

# Create a CSV file for storing the ball coordinates
csv_file = open('locations.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['X', 'Y'])
posList = []
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

frame_count = 0
last_ball_coordinates = None

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    ball_coordinates = detect_ball_coordinates(frame, last_ball_coordinates)

    # Draw the ball coordinates on the frame
    if ball_coordinates is not None:
        x, y = ball_coordinates
        posList.append(ball_coordinates)

        # Write the ball coordinates to the CSV file
        csv_writer.writerow([x, 1080 - y])

        last_ball_coordinates = ball_coordinates

    for i, pos in enumerate(posList):
        cv2.circle(frame, pos, 3, (0, 255, 0), -1)
        if i == 0:
            cv2.line(frame, pos, pos, (0, 0, 255), 2)
        else:
            cv2.line(frame, pos, posList[i - 1], (0, 0, 255), 2)

    # Display the frame with ball coordinates
    cv2.imshow('Video', frame)
    out.write(frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
csv_file.close()
cv2.destroyAllWindows()
