import os
import shutil
import random
from sklearn.model_selection import train_test_split


base_dir = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Kidney Cancer/' # Path to your original dataset
categories = ['kidney_normal', 'kidney_tumor']

# Paths for train/test splits
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Create directories if they don't exist
for split_dir in [train_dir, test_dir]:
    for category in categories:
        os.makedirs(os.path.join(split_dir, category), exist_ok=True)

# Split ratio
test_ratio = 0.2  # 20% for test data

for category in categories:
    folder = os.path.join(base_dir, category)
    images = os.listdir(folder)
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Train-test split
    train_images, test_images = train_test_split(images, test_size=test_ratio, random_state=42)

    # Copy images
    for img in train_images:
        shutil.move(os.path.join(folder, img), os.path.join(train_dir, category, img))
    for img in test_images:
        shutil.move(os.path.join(folder, img), os.path.join(test_dir, category, img))

print("Dataset successfully split into train and test folders.")

