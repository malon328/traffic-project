import cv2
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Load image
image_path = "output/violation_test.jpg"

img = cv2.imread(image_path)

# Run OCR
results = reader.readtext(img)

print("\nDetected Text:\n")

for result in results:
    text = result[1]
    confidence = result[2]

    print(f"Text: {text}")
    print(f"Confidence: {confidence}")
    print("-------------------")