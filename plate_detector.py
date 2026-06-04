import cv2
import easyocr
from ultralytics import YOLO

# Load plate detection model
model = YOLO("models/license_plate_detector.pt")

# Initialize OCR
reader = easyocr.Reader(['en'])

# Load image
image_path = "output/violation_test.jpg"
img = cv2.imread(image_path)

# Detect license plates
results = model(img)

for result in results:
    boxes = result.boxes

    for box in boxes:

        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop plate
        plate_crop = img[y1:y2, x1:x2]

        # -------------------------
        # PREPROCESS PLATE IMAGE
        # -------------------------

        # Resize plate (important)
        plate_crop = cv2.resize(plate_crop, None, fx=3, fy=3)

        # Convert to grayscale
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

        # Reduce noise
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # OCR on grayscale image
        ocr_results = reader.readtext(gray)

        # Show processed plate
        cv2.imshow("Processed Plate", gray)

        print("\nDetected Plate Text:\n")

        for ocr in ocr_results:
            text = ocr[1]
            confidence = ocr[2]

            print(f"Plate: {text}")
            print(f"Confidence: {confidence}")

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

# Show result
cv2.imshow("Plate Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()