import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import easyocr
import re
import threading

app = Flask(__name__)
CORS(app)

# ============================================================
#  PRE-LOAD EVERYTHING AT STARTUP (not on first request)
# ============================================================
print("⚡ Loading models into RAM...")

# Use 'yolov8n.pt' or ideally a plate-specific model like:
# 'keremberke/yolov8n-license-plate-detection' (Hugging Face)
# To use it: model = YOLO('keremberke/yolov8n-license-plate-detection')
model = YOLO('yolov8n.pt')

# Warm up the model with a dummy image (first inference is always slow)
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
model(dummy, verbose=False)

# EasyOCR - pre-load into memory with GPU if available
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

# Pre-warm EasyOCR too
reader.readtext(np.zeros((50, 200), dtype=np.uint8))

print("✅ All models warmed up. Server ready for <0.1s inference!")

# ============================================================
#  INDIAN PLATE REGEX (post-processing filter)
# ============================================================
INDIAN_PLATE_PATTERN = re.compile(
    r'^[A-Z]{2}[\s]?\d{1,2}[\s]?[A-Z]{1,3}[\s]?\d{1,4}$'
)

def format_indian_plate(text):
    """Clean and format Indian plate number."""
    # Remove common misreads
    text = text.upper().strip()
    text = text.replace('IND', '').replace('BH ', '').strip()
    # Keep only alphanumeric
    text = re.sub(r'[^A-Z0-9]', '', text)
    # Common OCR error corrections for Indian plates
    corrections = {'O': '0', 'I': '1', 'S': '5', 'B': '8'}
    # Only fix digits in numeric positions (positions 2-4)
    if len(text) >= 4:
        corrected = list(text)
        # Positions 2,3 should be digits (district code)
        for i in [2, 3]:
            if i < len(corrected) and corrected[i] in corrections:
                corrected[i] = corrections[corrected[i]]
        text = ''.join(corrected)
    return text

# ============================================================
#  ULTRA-FAST IMAGE PREPROCESSING PIPELINE
# ============================================================
def preprocess_for_ocr(crop):
    """Apply fast image enhancement for better OCR accuracy."""
    # Resize to standard height for consistent OCR
    h, w = crop.shape[:2]
    target_h = 80  # optimal for EasyOCR speed vs accuracy
    scale = target_h / h
    new_w = int(w * scale)
    resized = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    # Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # CLAHE for contrast (faster than full histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    
    # Otsu threshold (auto binary)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

# ============================================================
#  MAIN SCAN ENDPOINT
# ============================================================
@app.route('/scan', methods=['POST'])
def scan_plate():
    if 'image' not in request.files:
        return jsonify({"plate": "Not Found", "status": "Failed", "error": "No image"})
    
    # --- Step 1: Decode image (fast numpy path) ---
    file_bytes = request.files['image'].read()
    npimg = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"plate": "Not Found", "status": "Failed"})

    # --- Step 2: Downscale for YOLO (speed trick) ---
    # YOLO works well at 320px for plate detection — 2x faster than 640px
    h, w = img.shape[:2]
    yolo_img = cv2.resize(img, (320, 320)) if max(h, w) > 320 else img
    
    # --- Step 3: YOLO Detection ---
    results = model(yolo_img, conf=0.3, iou=0.45, imgsz=320, verbose=False)
    
    plate_text = "Not Found"
    best_conf = 0

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
            
        # Scale boxes back to original image size
        scale_x = w / 320
        scale_y = h / 320
        
        for i, box in enumerate(r.boxes.xyxy):
            conf = float(r.boxes.conf[i]) if r.boxes.conf is not None else 1.0
            if conf < best_conf:
                continue
            
            x1, y1, x2, y2 = (int(box[0] * scale_x), int(box[1] * scale_y),
                               int(box[2] * scale_x), int(box[3] * scale_y))
            
            # Add padding
            pad = 8
            crop = img[max(0, y1-pad):min(h, y2+pad),
                       max(0, x1-pad):min(w, x2+pad)]
            
            if crop.size == 0:
                continue
            
            # --- Step 4: Preprocess crop ---
            processed = preprocess_for_ocr(crop)
            
            # --- Step 5: EasyOCR with tight config for speed ---
            ocr_res = reader.readtext(
                processed,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                batch_size=1,
                detail=1,
                paragraph=False,
                min_size=10
            )
            
            if ocr_res:
                # Combine all detected text segments
                all_text = ''.join([r[1] for r in ocr_res])
                formatted = format_indian_plate(all_text)
                
                if len(formatted) >= 6:  # Minimum valid plate length
                    plate_text = formatted
                    best_conf = conf

    # --- Step 6: Fallback — if YOLO found nothing, try full-image OCR ---
    if plate_text == "Not Found":
        # Crop center-bottom area where plates usually are
        center_crop = img[int(h*0.5):h, int(w*0.1):int(w*0.9)]
        if center_crop.size > 0:
            processed_full = preprocess_for_ocr(center_crop)
            ocr_res = reader.readtext(
                processed_full,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                detail=1, paragraph=False
            )
            if ocr_res:
                best = max(ocr_res, key=lambda x: len(x[1]))
                formatted = format_indian_plate(best[1])
                if len(formatted) >= 6:
                    plate_text = formatted

    return jsonify({
        "plate": plate_text,
        "status": "Success" if plate_text != "Not Found" else "Failed"
    })

# ============================================================
#  HEALTH CHECK ENDPOINT
# ============================================================
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "alive"})

if __name__ == '__main__':
    print(f"\n🚀 Ultra-Fast ANPR Server running at http://0.0.0.0:5501")
    print(f"📱 For mobile, use your PC's IP: http://YOUR_IP:5501/scan\n")
    # threaded=True handles multiple requests simultaneously
    app.run(host='0.0.0.0', port=5501, threaded=True, debug=False)