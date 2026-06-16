import os
import pandas as pd
from ultralytics import YOLO

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
model_path = os.path.join(PROJECT_ROOT, "runs/obb/sentinel-kiln-repro/yolov8n-obb_20260611_162305/weights/best.pt")
image_dir = os.path.join(PROJECT_ROOT, "sentinel/palwal/rgb")
output_csv = os.path.join(PROJECT_ROOT, "data/detections.csv")

def run_inference():
    if not os.path.exists(model_path):
        print(f"CRITICAL ERROR: Model weights not found at '{model_path}'.")
        return
        
    if not os.path.exists(image_dir) or len(os.listdir(image_dir)) == 0:
        print(f"CRITICAL ERROR: No image patches found in '{image_dir}'.")
        return

    print("Loading YOLO8 Nano model...")
    model = YOLO(model_path)

    print(f"Scanning patches in {image_dir} for brick kilns...")
    # stream=True keeps memory usage low by processing one image at a time
    results = model(image_dir, stream=True, verbose=False)

    detections = []
    
    for r in results:
        img_name = os.path.basename(r.path)
        boxes = r.obb if r.obb is not None else r.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].tolist()
            
            # --- NEW CODE: Grab the predicted class ---
            cls_id = int(box.cls[0].item())
            predicted_class = model.names[cls_id]
            # ------------------------------------------
            
            detections.append({
                "image_name": img_name,
                "predicted_class": predicted_class,  # Add this to the CSV
                "confidence": round(conf, 4),
                "x_min": round(x1, 1),
                "y_min": round(y1, 1),
                "x_max": round(x2, 1),
                "y_max": round(y2, 1)
            })

    # Export results
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(detections)
    df.to_csv(output_csv, index=False)
    
    print(f"\nSUCCESS: Inference complete.")
    print(f"Found {len(detections)} potential kilns.")
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    run_inference()
