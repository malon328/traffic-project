import os
import json
import requests
import cv2
import easyocr
import cloudinary
import cloudinary.uploader

# ==============================================================================
# CONFIGURATION
# ==============================================================================
cloudinary.config( 
    cloud_name = "db1ujrt64", 
    api_key = "434877477258887", 
    api_secret = "CLgs7LX2AVjwqFUu1-8LELA-Y68",
    secure = True
)

DASHBOARD_ENDPOINT = "https://traffic-management-xhp7.onrender.com/api/v1/violations"

print("[SYSTEM] Pre-warming background execution EasyOCR components...")
reader = easyocr.Reader(['en'], gpu=False)

def handle_violation_pipeline(image_path, plate_text, iso_timestamp, bbox):
    if not os.path.exists(image_path):
        return

    print(f"\n[PIPELINE START] Processing background worker job for: {image_path}")
    
    # 1. OCR CORRELATION
    img = cv2.imread(image_path)
    if img is not None:
        x1, y1, x2, y2 = bbox
        h, w, _ = img.shape
        pad_y1, pad_y2 = max(0, y1-10), min(h, y2+10)
        pad_x1, pad_x2 = max(0, x1-10), min(w, x2+10)
        car_crop = img[pad_y1:pad_y2, pad_x1:pad_x2]
        results = reader.readtext(car_crop, detail=0)
        detected_plate = (
            "".join(results)
            .replace(" ", "")
            .replace("'", "")
            .replace('"', "")
            .upper()
            if results
            else "UNKNOWN"
        )
    else:
        detected_plate = "UNKNOWN"

    print(f"[OCR RESULT] Target Identified: {detected_plate}")

    # 2. CLOUDINARY UPLOAD PROTOCOL
    print(f"[CLOUDINARY] Pushing asset directly to cloud hosting...")
    try:
        # Standard folder path mapping inside your cloud storage container
        upload_result = cloudinary.uploader.upload(
            image_path, 
            folder="traffic/violations",
            public_id=os.path.basename(image_path).split('.')[0]
        )
        
        # Extract the permanent secure URL link string
        cloud_image_url = upload_result.get("secure_url")
        print(f"[CLOUDINARY SUCCESS] Image live on CDN: {cloud_image_url}")

        # 3. SCHEMA BUILDING & LOG PRINT
        payload = {
            "plateNumber": detected_plate,
            "violationType": "RED_LIGHT",
            "imageUrls": [cloud_image_url],
            "violationAt": iso_timestamp,
            "status": "PENDING"
        }

        print("\n" + "="*50)
        print("    LIVE CLOUD LOG OUTPUT")
        print("="*50)
        print(json.dumps(payload, indent=4))
        print("="*50 + "\n")

        print(f"[DASHBOARD API] Forwarding packet data schema map to database...")
        dashboard_response = requests.post(DASHBOARD_ENDPOINT, json=payload, timeout=20)
        if dashboard_response.status_code in [200, 201]:
            print("[PIPELINE COMPLETE] Event logged cleanly in tracker table space.")
        else:
            print(f"[API WARN] System received non-success response code: {dashboard_response.status_code}")
            print(f"Details: {dashboard_response.text}")

    except Exception as e:
        print(f"[PIPELINE ERROR] Cloud execution failed: {e}")