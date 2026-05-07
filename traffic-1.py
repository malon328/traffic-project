import cv2
from ultralytics import YOLO
import time
import os
import numpy as np


# -----------------------------
# TRAFFIC LIGHT DETECTION
# -----------------------------
def detect_traffic_light(frame):
    h, w, _ = frame.shape

    # TOP-RIGHT REGION
    roi = frame[
        int(h * 0.096):int(h * 0.19),
        int(w * 0.51):int(w * 0.56)
    ]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # RED MASK
    lower_red1 = (0, 120, 70)
    upper_red1 = (10, 255, 255)

    lower_red2 = (170, 120, 70)
    upper_red2 = (180, 255, 255)

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + \
               cv2.inRange(hsv, lower_red2, upper_red2)

    # GREEN MASK
    lower_green = (40, 40, 40)
    upper_green = (90, 255, 255)

    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)

    if red_pixels > green_pixels:
        return "RED"

    elif green_pixels > red_pixels:
        return "GREEN"

    else:
        return "UNKNOWN"


# -----------------------------
# POLYGON CHECK
# -----------------------------
def is_inside_polygon(cx, cy, polygon):
    return cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0


# -----------------------------
# SETUP
# -----------------------------
os.makedirs("output-1", exist_ok=True)

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("traffic-1.mp4")

if not cap.isOpened():
    print("ERROR: Could not open video file")
    exit()


# -----------------------------
# MAIN LOOP
# -----------------------------
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    # -----------------------------
    # BLUE REGION POLYGON
    # -----------------------------
    lane_polygon = np.array([
        [int(-w * 0.7), int(h * 0.9)],  # bottom-left
        [w, int(h * 0.9)],  # bottom-right
        [w, int(h * 0.6)],  # top-right
        [int(w * 0.2), int(h * 0.6)]   # top-left
    ], np.int32)

    # Detect traffic light
    traffic_light = detect_traffic_light(frame)

    # -----------------------------
    # DRAW TRAFFIC LIGHT ROI
    # -----------------------------
    cv2.rectangle(
        frame,
        (int(w * 0.51), int(h * 0.096)),
        (int(w * 0.56), int(h * 0.19)),
        (0, 255, 255),
        2
    )

    # -----------------------------
    # DRAW BLUE POLYGON AREA
    # -----------------------------
    overlay = frame.copy()

    cv2.fillPoly(overlay, [lane_polygon], (255, 0, 0))

    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    cv2.polylines(
        frame,
        [lane_polygon],
        isClosed=True,
        color=(255, 0, 0),
        thickness=3
    )

    # -----------------------------
    # STOP LINE
    # -----------------------------
    lx1 = 0
    ly1 = int(h * (1 - 1/4))

    lx2 = w
    ly2 = int(h * (1 - 1/4))

    cv2.line(frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 3)

    # -----------------------------
    # TRAFFIC LIGHT STATUS
    # -----------------------------
    cv2.putText(
        frame,
        f"Light: {traffic_light}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # -----------------------------
    # YOLO DETECTION
    # -----------------------------
    results = model(frame)

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])

            # CLASS 2 = CAR
            if cls == 2:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # CENTER POINT
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # -----------------------------
                # CHECK IF INSIDE BLUE REGION
                # -----------------------------
                if not is_inside_polygon(cx, cy, lane_polygon):
                    continue

                # DRAW CAR BOX
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # CENTER POINT
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

                # -----------------------------
                # STOP LINE POSITION
                # -----------------------------
                line_y_at_x = ly1 + (ly2 - ly1) * (cx - lx1) / (lx2 - lx1)

                # -----------------------------
                # VIOLATION CHECK
                # -----------------------------
                if traffic_light == "RED" and cy < line_y_at_x:

                    cv2.putText(
                        frame,
                        "VIOLATION!",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    timestamp = int(time.time())

                    cv2.imwrite(
                        f"output-1/violation_{timestamp}.jpg",
                        frame
                    )

                    print(
                        f"Violation detected at {timestamp} - Plate: ABC123"
                    )

    # -----------------------------
    # SHOW OUTPUT
    # -----------------------------
    cv2.imshow("Traffic AI System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()