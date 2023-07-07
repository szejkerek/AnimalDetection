import os
import random
import string
import shutil

# Set the paths to the folders
images_folder = "Images/"
masks_folder = "Masks/"
images_renamed_folder = "ImagesRenamed/"
masks_renamed_folder = "MasksRenamed/"

# Create the renamed folders if they don't exist
os.makedirs(images_renamed_folder, exist_ok=True)
os.makedirs(masks_renamed_folder, exist_ok=True)

# Get the list of files in the Images folder
image_files = os.listdir(images_folder)

# Shuffle the list of files randomly
random.shuffle(image_files)


# Generate a random string of lowercase letters and digits
def generate_random_name(length=8):
    letters_digits = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters_digits) for _ in range(length))


# Rename the files in both folders
for i, image_file in enumerate(image_files):
    # Get the corresponding mask file name
    image_name = os.path.splitext(image_file)[0]
    mask_file = next((f for f in os.listdir(masks_folder) if os.path.splitext(f)[0] == image_name), None)

    if mask_file:
        # Generate random names for both files
        new_name = generate_random_name()

        # Get the file extensions
        image_extension = os.path.splitext(image_file)[1]
        mask_extension = os.path.splitext(mask_file)[1]

        # Generate the new file paths with random names and extensions
        new_image_path = os.path.join(images_renamed_folder, new_name + image_extension)
        new_mask_path = os.path.join(masks_renamed_folder, new_name + mask_extension)

        # Rename and move the files
        shutil.move(os.path.join(images_folder, image_file), new_image_path)
        shutil.move(os.path.join(masks_folder, mask_file), new_mask_path)

        print(f"Renamed and moved: {image_file} -> {new_name + image_extension}")
        print(f"Renamed and moved: {mask_file} -> {new_name + mask_extension}")
    else:
        print(f"No corresponding mask found for {image_file}")
