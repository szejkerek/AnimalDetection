"""\
Validating dataset
Author: Bartłomiej Gordon
"""

import os
from collections import namedtuple
import cv2
from PIL import Image

def is_mask_pixel(pixel):
    red, green, blue = pixel[:3]
    return (red == 255 and green == 0 and blue == 0) or \
        (red == 0 and green == 255 and blue == 0) or \
        (red == 0 and green == 0 and blue == 255) or \
        (red == 255 and green == 255 and blue == 255) or \
        (red == 0 and green == 0 and blue == 0)
def is_blue(pixel):
    red, green, blue = pixel[:3]
    return (red == 0 and green == 0 and blue == 255)
def is_green(pixel):
    red, green, blue = pixel[:3]
    return (red == 0 and green == 255 and blue == 0)
def is_red(pixel):
    red, green, blue = pixel[:3]
    return (red == 255 and green == 0 and blue == 0)



def check_pixels(mask_path, pure_color_treshold, no_animal_treshold):
    mask_image = Image.open(mask_path)
    mask_pixels = mask_image.load()

    total_pixels = mask_image.width * mask_image.height
    count_pure_color = 0
    blue_pixels = 0
    red_pixels = 0
    green_pixels = 0

    for x in range(mask_image.width):
        for y in range(mask_image.height):
            if is_mask_pixel(mask_pixels[x, y]):
                count_pure_color += 1
            if is_blue(mask_pixels[x,y]):
                blue_pixels += 1
            if is_green(mask_pixels[x,y]):
                green_pixels += 1
            if is_red(mask_pixels[x,y]):
                red_pixels += 1


    percentage_pure_color = count_pure_color / total_pixels
    percentage_animal_pixels = blue_pixels / total_pixels

    mask_is_not_pure_color = not percentage_pure_color > pure_color_treshold
    no_animal = percentage_animal_pixels < no_animal_treshold
    swapped_colors = blue_pixels > green_pixels

    return mask_is_not_pure_color, no_animal, swapped_colors


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


def get_file_names_with_extensions(folder_path):
    files = os.listdir(folder_path)
    return [file for file in files if os.path.isfile(os.path.join(folder_path, file))]


def find_matching_pairs(images_folder, masks_folder):
    ImagePair = namedtuple('ImagePair', ['image_path', 'mask_path'])
    image_files = get_file_names_with_extensions(images_folder)
    mask_files = get_file_names_with_extensions(masks_folder)
    image_names = [os.path.splitext(file)[0] for file in image_files]
    matching_pairs = []

    for mask_file in mask_files:
        mask_name = os.path.splitext(mask_file)[0]
        if mask_name in image_names:
            image_index = image_names.index(mask_name)
            image_path = os.path.join(images_folder, image_files[image_index])
            mask_path = os.path.join(masks_folder, mask_file)
            matching_pair = ImagePair(image_path, mask_path)
            matching_pairs.append(matching_pair)

    return matching_pairs


def get_file_names(folder_path):
    files = os.listdir(folder_path)
    return [os.path.splitext(file)[0] for file in files]


def find_missing_images(images_folder, masks_folder):
    images_names = get_file_names(images_folder)
    masks_names = get_file_names(masks_folder)
    images_without_mask = set(images_names) - set(masks_names)
    masks_without_image = set(masks_names) - set(images_names)
    return images_without_mask, masks_without_image


images_folder = "Images"
masks_folder = "Masks"

images_without_mask, masks_without_image = find_missing_images(images_folder, masks_folder)

matching_pairs = find_matching_pairs(images_folder, masks_folder)

output_file = "report.txt"

with open(output_file, "w", encoding="utf-8") as file:
    file.write("\nImages without mask:\n")
    for image_name in images_without_mask:
        print("MasksVSImages")
        file.write(image_name + "\n")

    file.write("\nMasks without image:\n")
    for image_name in masks_without_image:
        print("MasksVSImages")
        file.write(image_name + "\n")

    file.write("\nMask and image are different size:\n")
    for pair in matching_pairs:
        print("Pairs")
        if not same_shape(pair.image_path, pair.mask_path):
            file.write(pair.image_path + " != " + pair.mask_path + "\n")

    file.write("\nCannot be opened using CV2 (polish symbols):\n")
    for pair in matching_pairs:
        print("CV2")
        if not check_image(pair.image_path):
            file.write(pair.image_path + "\n")
        if not check_image(pair.mask_path):
            file.write(pair.mask_path + "\n")

    pure_color_threshold = 0.95
    no_animal_threshold = 0.001
    file.write("\nPossible mask problems:\n")
    file.write("Mask colors are not pure - more than "+str(round((1-pure_color_threshold)*100, 2))+"% of image contains colors different than red/green/blue/white/black.\n")
    file.write("Animal too small or absent - animal cover less than "+str(round(no_animal_threshold*100, 2))+"% of image.\n")
    file.write("Colors are probably swapped - there is more blue than green.\n")
    file.write("\n")
    for pair in matching_pairs:
        print("Pixels")
        mask_is_not_pure_color, no_animal, swapped_colors = check_pixels(pair.mask_path,pure_color_threshold,no_animal_threshold)

        if mask_is_not_pure_color:
            file.write("Mask colors are not pure: " + pair.mask_path + "\n")

        if no_animal:
            file.write("Animal too small or absent: " + pair.mask_path + "\n")

        if swapped_colors:
            file.write("Colors are probably swapped: " + pair.mask_path + "\n")

        if mask_is_not_pure_color or no_animal or swapped_colors:
            file.write("\n")

print("Report saved to", output_file)
