import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

os.makedirs('workspace_results/presentation_assets/', exist_ok=True)

# 1. Input your actual mAP50 scores here once the runs finish
data = {
    'Model': ['YOLOv8n', 'YOLO11n'],
    'Data Split': ['5% Subset', '5% Subset'],
    'mAP50': [0.701  , 0.632 ] # <--- REPLACE WITH YOUR REAL NUMBERS
}

df = pd.DataFrame(data)

# 2. Setup the plot geometry
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.5
x = np.arange(len(df))  # One bar per model

# 3. Extract the 5% subset scores
scores = df['mAP50'].values
labels = df['Model'].values

rects = ax.bar(x, scores, bar_width, label='5% Subset', color='#1f77b4')

# 4. Formatting
ax.set_ylabel('mAP50 Score', fontsize=12, fontweight='bold')
ax.set_title('Model Architecture vs. Data Scaling on SentinelKilnDB', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['YOLOv8 Nano', 'YOLO11 Nano'], fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim(0, 1.0) # Keeps scale honest

# Add value labels on top of bars
ax.bar_label(rects, padding=3, fmt='%.3f')

# Save the asset
plt.tight_layout()
save_path = 'workspace_results/presentation_assets/mAP50_scaling_comparison.png'
plt.savefig(save_path, dpi=300)
print(f"Chart successfully saved to: {save_path}")