import cv2
from ultralytics import YOLO
import time
import os
import numpy as np
import threading
from datetime import datetime
# Import your modularized background pipeline network worker
from uploader import handle_violation_pipeline

# -----------------------------
# THREADED REAL-TIME STREAM VIDEO CAPTURE
# -----------------------------
class RTSPVideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            ret, frame = self.cap.read()
            if ret:
                with self.read_lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.read_lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()


# -----------------------------
# TRAFFIC LIGHT DETECTION
# -----------------------------
def detect_traffic_light(frame):
    h, w, _ = frame.shape
    roi = frame[int(h * 0.03):int(h * 0.15), int(w * 0.49):int(w * 0.545)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red1 = (0, 120, 70)
    upper_red1 = (10, 255, 255)
    lower_red2 = (170, 120, 70)
    upper_red2 = (180, 255, 255)
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    lower_green = (40, 40, 40)
    upper_green = (90, 255, 255)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    if cv2.countNonZero(red_mask) > cv2.countNonZero(green_mask):
        return "RED"
    elif cv2.countNonZero(green_mask) > cv2.countNonZero(red_mask):
        return "GREEN"
    return "RED"


def is_inside_polygon(cx, cy, polygon):
    return cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0


# -----------------------------
# INITIALIZATION SETUP
# -----------------------------
os.makedirs("output-1", exist_ok=True)
model = YOLO("yolov8n.pt")
stream_url = "http://192.168.43.34:81/stream" 

vs = RTSPVideoStream(stream_url).start()
time.sleep(1.0)

frame_count = 0
skip_frames = 3  
tracked_cars = []

last_violation_time = 0.0
violation_cooldown = 5.0  

# -----------------------------
# MAIN EVENT STREAM LOOP
# -----------------------------
while vs.isOpened():
    
    ret, frame = vs.read()
    if not ret or frame is None:
        continue

    frame_count += 1
    h, w, _ = frame.shape
    clean_frame_for_ocr = frame.copy()

    # Visual layout layout boundaries
    lane_polygon = np.array([
        [int(w * 0.12), int(h * 0.95)],    
        [int(w * 1.1), int(h * 0.95)],               
        [int(w * 0.75), int(h * 0.35)],   
        [int(w * 0.35), int(h * 0.35)]   
    ], np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [lane_polygon], (255, 0, 0))
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    cv2.polylines(frame, [lane_polygon], isClosed=True, color=(255, 0, 0), thickness=3)

    lx1, ly1, lx2, ly2 = 0, int(h * (1 - 1/2)), w, int(h * (1 - 1/2))
    cv2.line(frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 3)

    traffic_light = detect_traffic_light(clean_frame_for_ocr)
    cv2.rectangle(frame, (int(w * 0.495), int(h * 0.04)), (int(w * 0.545), int(h * 0.14)), (0, 255, 255), 2)
    cv2.putText(frame, f"Light: {traffic_light}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if frame_count % skip_frames != 0:
        for car in tracked_cars:
            tx1, ty1, tx2, ty2 = car['bbox']
            tcx, tcy = car['center']
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
            cv2.circle(frame, (tcx, tcy), 5, (0, 255, 255), -1)
            if car['violation']:
                cv2.putText(frame, "VIOLATION!", (tx1, ty1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Traffic AI System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # YOLO Inference execution
    tracked_cars = [] 
    results = model(clean_frame_for_ocr, verbose=False)
    current_time = time.time()

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 2: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if not is_inside_polygon(cx, cy, lane_polygon):
                    continue

                line_y_at_x = ly1 + (ly2 - ly1) * (cx - lx1) / (lx2 - lx1)
                is_violating = (traffic_light == "RED" and cy < line_y_at_x)

                tracked_cars.append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (cx, cy),
                    'violation': is_violating
                })

                if is_violating:
                    if (current_time - last_violation_time) > violation_cooldown:
                        last_violation_time = current_time
                        
                        # Generate precise ISO 8601 Zulu string
                        iso_timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                        img_path = f"output-1/violation_{int(current_time)}.jpg"
                        
                        cv2.imwrite(img_path, clean_frame_for_ocr)
                        print(f"\n[ALERT] Snapshot taken: {img_path}")
                        
                        # Pack coordinates and invoke the multi-threaded background pipeline handler
                        bbox_coords = (x1, y1, x2, y2)
                        threading.Thread(
                            target=handle_violation_pipeline, 
                            args=(img_path, "OCR_PENDING", iso_timestamp, bbox_coords)
                        ).start()

    for car in tracked_cars:
        tx1, ty1, tx2, ty2 = car['bbox']
        tcx, tcy = car['center']
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
        cv2.circle(frame, (tcx, tcy), 5, (0, 255, 255), -1)
        if car['violation']:
            cv2.putText(frame, "VIOLATION!", (tx1, ty1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Traffic AI System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()