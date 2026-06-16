import os, shutil, random
from glob import glob
from tqdm import tqdm

SOURCE_IMG_DIR = "data/raw/train/images"
SOURCE_LBL_DIR = "data/raw/train/yolo_obb_labels"
TARGET_DIR = "data/split"

if os.path.exists(TARGET_DIR): shutil.rmtree(TARGET_DIR)

for split in ['train', 'val']:
    os.makedirs(f"{TARGET_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{TARGET_DIR}/labels/{split}", exist_ok=True)

all_images = glob(f"{SOURCE_IMG_DIR}/*.png") + glob(f"{SOURCE_IMG_DIR}/*.jpg")
image_buckets = {'rare': [], 'common': [], 'majority': [], 'negative': []}

print("Scanning dataset...")
for img_path in tqdm(all_images):
    base_name = os.path.basename(img_path)
    lbl_path = os.path.join(SOURCE_LBL_DIR, base_name.rsplit('.', 1)[0] + '.txt')
    
    if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
        image_buckets['negative'].append((img_path, None))
        continue
        
    with open(lbl_path, 'r') as f:
        classes = set(int(line.split()[0]) for line in f if line.strip())
        
    if 0 in classes: image_buckets['rare'].append((img_path, lbl_path))
    elif 2 in classes: image_buckets['common'].append((img_path, lbl_path))
    else: image_buckets['majority'].append((img_path, lbl_path))

print("\nExtracting 5% subset...")
for bucket, items in image_buckets.items():
    random.shuffle(items)
    # Isolate exactly 5% of the data
    subset_size = int(len(items) * 0.10)  # 10% to ensure we have enough for train/val split
    subset_items = items[:subset_size]
    train_idx = int(len(subset_items) * 0.8)
    
    for i, (img, lbl) in enumerate(tqdm(subset_items, desc=f"Copying {bucket}")):
        split = 'train' if i < train_idx else 'val'
        shutil.copy(img, f"{TARGET_DIR}/images/{split}/{os.path.basename(img)}")
        if lbl: 
            shutil.copy(lbl, f"{TARGET_DIR}/labels/{split}/{os.path.basename(lbl)}")

with open(f'{TARGET_DIR}/dataset.yaml', 'w') as f:
    f.write(f"path: {os.path.abspath(TARGET_DIR)}\ntrain: images/train\nval: images/val\nnc: 3\nnames: ['CFCBK', 'FCBK', 'Zigzag']")
print("\nSubset staging complete.")