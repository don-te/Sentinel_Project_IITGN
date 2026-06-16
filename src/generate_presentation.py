from ultralytics import YOLO
import os
import shutil
import pandas as pd

ASSET_DIR = 'workspace_results/presentation_assets/'
os.makedirs(ASSET_DIR, exist_ok=True)

# 1. Update this dictionary with the exact paths to your 4 runs.
# Check your runs/obb/ directory to see which train folder corresponds to which run!
models = {
    "YOLO11n-OBB (5%)": "runs/obb/sentinel-kiln-repro/yolo11n-obb_20260610_143307/weights/best.pt",   # e.g., Run 1
    "YOLOv8n-OBB (5%)": "runs/obb/sentinel-kiln-repro/yolov8n-obb_20260611_162305/weights/best.pt",  # e.g., Run 2
}

overall_results = []
class_results = []

for name, weight_path in models.items():
    print(f"\nEvaluating {name}...")
    if not os.path.exists(weight_path):
        print(f"ERROR: Weights not found at {weight_path}. Check your folder paths.")
        continue

    # Load model and run validation on the 20% validation split
    model = YOLO(weight_path)
    metrics = model.val(data="data/split/dataset.yaml", split="val", plots=True)

    # Extract Overall Metrics
    overall_results.append({
        "Model": name,
        "mAP50": round(metrics.box.map50, 4),
        "mAP50-95": round(metrics.box.map, 4),
        "Precision": round(metrics.box.p.mean(), 4),
        "Recall": round(metrics.box.r.mean(), 4)
    })

    # Extract Class-Wise mAP50
    for i, class_name in metrics.names.items():
        if i < len(metrics.box.ap50):
            class_results.append({
                "Model": name,
                "Class": class_name,
                "mAP50": round(metrics.box.ap50[i], 4)
            })

    # 2. Extract and Copy Visual Assets
    val_dir = metrics.save_dir
    plots_to_copy = {
        'confusion_matrix.png': f'{name}_confusion_matrix.png',
        'PR_curve.png': f'{name}_PR_curve.png',
        'F1_curve.png': f'{name}_F1_curve.png',
        'val_batch0_pred.jpg': f'{name}_qualitative_predictions.jpg' # Qualitative overlay grid
    }

    for src_file, dest_file in plots_to_copy.items():
        src_path = os.path.join(val_dir, src_file)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(ASSET_DIR, dest_file))

# 3. Generate Markdown Tables
df_overall = pd.DataFrame(overall_results)
df_class = pd.DataFrame(class_results).pivot(index='Model', columns='Class', values='mAP50').reset_index()

with open(os.path.join(ASSET_DIR, "quantitative_results.md"), "w") as f:
    f.write("# Model Comparison: YOLO11n vs YOLOv8n (5% Subset)\n\n")
    f.write("## Overall Metrics\n")
    f.write(df_overall.to_markdown(index=False))
    f.write("\n\n## Class-Wise mAP50 Breakdown\n")
    f.write(df_class.to_markdown(index=False))

print(f"\nAll presentation assets have been successfully saved to '{ASSET_DIR}'.")
print("You can now download the images and markdown tables directly via VS Code.")