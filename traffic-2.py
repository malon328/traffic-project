import cv2
from ultralytics import YOLO
import time
import os

# -----------------------------
# TRAFFIC LIGHT DETECTION
# -----------------------------
def detect_traffic_light(frame):
    h, w, _ = frame.shape

    # TOP-RIGHT REGION (your requirement)
    roi = frame[int(h * 0.2):int(h * 0.4), int(w * 0.2):int(w * 0.4)]

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
# DIAGONAL LANE CHECK
# bottom-left → top-right
# -----------------------------
def is_in_lane(x1, x2, y1, y2, frame):
    h, w, _ = frame.shape

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    nx = cx / w
    ny = cy / h

    # diagonal band (thickness control = 0.25)
    lower = (1 - nx) - 0.25
    upper = (1 - nx) + 0.25

    return lower < ny < upper


# -----------------------------
# SETUP
# -----------------------------
os.makedirs("output", exist_ok=True)

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("traffic-2.mp4")

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

    # Detect traffic light per frame
    traffic_light = detect_traffic_light(frame)

    # Draw traffic light ROI
    cv2.rectangle(frame, (int(w * 0.2), int(h * 0.1)), (int(w * 0.4), int(h * 0.4)), (0, 255, 255), 2)

    # Draw diagonal lane visualization
    cv2.line(frame, (int(w * 0.4), h), (int(w * 0.55), 0), (255, 0, 0), 2)

    # STOP LINE (adjusted right side lower)
    lx1 = 0
    ly1 = int(h * (1 - 2/5))

    lx2 = w
    ly2 = int(h * (1 - 1/4))

    cv2.line(frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 3)

    # Show traffic light status
    cv2.putText(frame, f"Light: {traffic_light}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # YOLO detection
    results = model(frame)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])

            # car class
            if cls == 2:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # only one diagonal lane
                if not is_in_lane(x1, x2, y1, y2, frame):
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # compute stop-line intersection
                line_y_at_x = ly1 + (ly2 - ly1) * (cx - lx1) / (lx2 - lx1)

                # violation logic
                if traffic_light == "RED" and cy > line_y_at_x:
                    cv2.putText(frame, "VIOLATION!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    timestamp = int(time.time())
                    cv2.imwrite(f"output/violation_{timestamp}.jpg", frame)

                    print(f"Violation detected at {timestamp} - Plate: ABC123")

    # show output
    cv2.imshow("Traffic AI System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()