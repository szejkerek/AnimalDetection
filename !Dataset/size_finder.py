import os
from PIL import Image

folder_path = "Masks"

total_width = 0
total_height = 0
num_images = 0

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        with Image.open(image_path) as img:
            width, height = img.size
            total_width += width
            total_height += height
            num_images += 1

if num_images == 0:
    print("No images found in the folder")
else:
    avg_width = total_width / num_images
    avg_height = total_height / num_images
    print(f"Average image shape (width, height) in {folder_path}: ({avg_width}, {avg_height})")
