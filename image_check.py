import pandas as pd
from PIL import Image
import os
from collections import Counter
import shutil

# Configuration
csv_path = "/teamspace/studios/this_studio/liwterm/data/pad-ufes-20_parsed_folders_train.csv"  # Absolute path
image_dir = "/teamspace/studios/this_studio/liwterm/data/imgs/"  # Absolute path
metadata_test_path = "/teamspace/studios/this_studio/liwterm/data/pad-ufes-20_parsed_test.csv"  # Not used currently

# Debug: Print working directory
print("Current working directory:", os.getcwd())

# Check if CSV exists
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at: {csv_path}")

# Read metadata CSV
df = pd.read_csv(csv_path)

# Function to get image size
def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

# Check sizes of all images
image_sizes = []
for img_id in df['img_id']:
    img_path = os.path.join(image_dir, img_id)
    if os.path.exists(img_path):
        size = get_image_size(img_path)
        if size:
            image_sizes.append(size)
    else:
        print(f"Image not found: {img_path}")

# Summarize sizes
print(f"Total images found - {len(image_sizes)}")
if image_sizes:
    size_counts = Counter(image_sizes)
    most_common_size, count = size_counts.most_common(1)[0]
    print(f"Most common image size: {most_common_size} with {count} images")
    print(f"Unique sizes found: {size_counts}")
else:
    print("No valid image sizes found.")

# Optional: Resize images to the most common size (uncomment to enable)
"""
output_dir = "/teamspace/studios/this_studio/liwterm/data/resized_imgs/"
backup_dir = "/teamspace/studios/this_studio/liwterm/data/backup_imgs/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(backup_dir, exist_ok=True)

if len(size_counts) > 1:
    print("Resizing images to", most_common_size)
    for img_id in df['img_id']:
        src_path = os.path.join(image_dir, img_id)
        dst_path = os.path.join(backup_dir, img_id)
        output_path = os.path.join(output_dir, img_id)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)  # Backup
            with Image.open(src_path) as img:
                if img.size != most_common_size:
                    img = img.resize(most_common_size, Image.Resampling.LANCZOS)
                    img.save(output_path, quality=95)
                    print(f"Resized {img_id} to {most_common_size}")
                else:
                    shutil.copy(src_path, output_path)
    df['img_path'] = df['img_id'].apply(lambda x: os.path.join(output_dir, x))
    df.to_csv("/teamspace/studios/this_studio/liwterm/data/updated_pad-ufes-20_parsed_folders_train.csv", index=False)
    print("Updated metadata saved.")
else:
    print("All images already have the same size:", most_common_size)
"""