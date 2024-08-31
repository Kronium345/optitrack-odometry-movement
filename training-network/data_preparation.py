import os
import shutil
import random
import glob

# 2.1 Define your dataset paths
dataset_path = './dataset_path/'
images_path = os.path.join(dataset_path, 'images')
labels_path = os.path.join(dataset_path, 'labels')

# Create directories for training and validation data
os.makedirs(os.path.join(images_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(images_path, 'val'), exist_ok=True)
os.makedirs(os.path.join(labels_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(labels_path, 'val'), exist_ok=True)

# 2.2 Split data into training and validation sets
# Define split ratio (e.g., 80% training, 20% validation)
train_ratio = 0.8

# List all image files
all_images = glob.glob(os.path.join(images_path, '*.jpg'))
random.shuffle(all_images)

# Calculate split index
split_index = int(len(all_images) * train_ratio)

# Split images into training and validation sets
train_images = all_images[:split_index]
val_images = all_images[split_index:]

# Function to move files to their respective folders
def move_files(file_list, dest_img_dir, dest_lbl_dir):
    for img_file in file_list:
        # Move image file
        shutil.move(img_file, dest_img_dir)
        # Move corresponding label file
        label_file = img_file.replace('images', 'labels').replace('.jpg', '.txt')
        shutil.move(label_file, dest_lbl_dir)

# Move training data
move_files(train_images, os.path.join(images_path, 'train'), os.path.join(labels_path, 'train'))

# Move validation data
move_files(val_images, os.path.join(images_path, 'val'), os.path.join(labels_path, 'val'))

print("Data preparation complete. Training and validation datasets are ready.")
