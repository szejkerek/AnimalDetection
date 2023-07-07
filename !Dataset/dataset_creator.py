import glob
import os
import random
import shutil
import sys
from pathlib import Path
import cv2
from PIL import Image

def same_shape(image_path1, image_path2):
    with Image.open(image_path1) as image1:
        with Image.open(image_path2) as image2:
            return image1.size == image2.size

def check_image(path):
    if not os.path.isfile(path):
        return False

    try:
        img = cv2.imread(path)
        if img is None:
            return False
    except cv2.error:
        return False

    return True


images_folder = "Images"
masks_folder = "Masks"
DIR_NAME = input("New dir name: ")

if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)
else:
    print("Folder already exists.")
    sys.exit()


matching_pairs = []
# Get a list of all mask filenames (without extensions)
masks_filenames = [Path(filename).stem for filename in os.listdir(masks_folder)]

# Iterate over the images folder and check if each image has a matching mask
for filename in os.listdir(images_folder):
    # Get the filename without extension
    image_filename = Path(filename).stem

    # Check if there is a matching mask
    if image_filename in masks_filenames:
        # Add the matching pair to the list
        matching_pairs.append(image_filename)
    else:
        print("Not maching: ", image_filename)

random.shuffle(matching_pairs)

# Split the matching pairs into training, validation, and testing sets
num_pairs = len(matching_pairs)
num_train = int(num_pairs * 0.7)
num_valid = int(num_pairs * 0.15)
num_test = num_pairs - num_train - num_valid

train_pairs = matching_pairs[:num_train]
valid_pairs = matching_pairs[num_train:num_train+num_valid]
test_pairs = matching_pairs[num_train+num_valid:]

# Create the output directories
train_images_dir = os.path.join("Train", "Images")
train_masks_dir = os.path.join("Train", "Masks")
valid_images_dir = os.path.join("Valid", "Images")
valid_masks_dir = os.path.join("Valid", "Masks")
test_images_dir = os.path.join("Test", "Images")
test_masks_dir = os.path.join("Test", "Masks")

for directory in [train_images_dir, train_masks_dir, valid_images_dir, valid_masks_dir, test_images_dir, test_masks_dir]:
    new_dir = os.path.join(DIR_NAME, directory)
    os.makedirs(new_dir, exist_ok=True)

for filename in train_pairs:
    image_files = glob.glob(os.path.join(images_folder, f"{filename}.*"))
    mask_files = glob.glob(os.path.join(masks_folder, f"{filename}.*"))
    image_file = image_files[0] if image_files else None
    mask_file = mask_files[0] if mask_files else None

    if check_image(image_file) and check_image(mask_file) and same_shape(image_file, mask_file):
        shutil.copy(image_file, os.path.join(DIR_NAME, train_images_dir))
        shutil.copy(mask_file, os.path.join(DIR_NAME, train_masks_dir))

# Copy the validation images and masks
for filename in valid_pairs:
    image_files = glob.glob(os.path.join(images_folder, f"{filename}.*"))
    mask_files = glob.glob(os.path.join(masks_folder, f"{filename}.*"))
    image_file = image_files[0] if image_files else None
    mask_file = mask_files[0] if mask_files else None
    if check_image(image_file) and check_image(mask_file) and same_shape(image_file, mask_file):
        shutil.copy(image_file, os.path.join(DIR_NAME, valid_images_dir))
        shutil.copy(mask_file, os.path.join(DIR_NAME, valid_masks_dir))

# Copy the testing images and masks
for filename in test_pairs:
    image_files = glob.glob(os.path.join(images_folder, f"{filename}.*"))
    mask_files = glob.glob(os.path.join(masks_folder, f"{filename}.*"))
    image_file = image_files[0] if image_files else None
    mask_file = mask_files[0] if mask_files else None
    if check_image(image_file) and check_image(mask_file) and same_shape(image_file, mask_file):
        shutil.copy(image_file, os.path.join(DIR_NAME, test_images_dir))
        shutil.copy(mask_file, os.path.join(DIR_NAME, test_masks_dir))


