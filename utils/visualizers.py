import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config
import utils
from utils import config_line


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def save_visualization(dataset, folder_name, enabled=False):
    if (enabled == False):
        return
    utils.create_subfolder_in_date_folder("vizualization_of_data_set")
    utils.create_subfolder_in_date_folder("vizualization_of_data_set\\"+folder_name)
    images_count = len(dataset)
    for n in range(images_count):
        image, mask = dataset[n]

        normal_image = Image.fromarray(np.uint8(image))
        animals = Image.fromarray(np.uint8(mask[..., 0].squeeze()))
        masking_background = Image.fromarray(np.uint8(mask[..., 1].squeeze()))
        nonmasking_background = Image.fromarray(np.uint8(mask[..., 2].squeeze()))
        foreground_attention = Image.fromarray(np.uint8(mask[..., 3].squeeze()))

        normal_image = normal_image.resize((normal_image.width * 2, normal_image.height * 2))
        new_image = Image.new('RGB', (normal_image.width * 2, normal_image.height))

        new_image.paste(normal_image, (0, 0))

        new_image.paste(animals, (normal_image.width, 0))
        new_image.paste(foreground_attention, (int(normal_image.width + normal_image.width/2), 0))

        new_image.paste(masking_background, (normal_image.width, int(normal_image.height/2)))
        new_image.paste(nonmasking_background, (int(normal_image.width + normal_image.width/2), int(normal_image.height/2)))

        path = os.path.join(config.CURRENT_PATH, os.path.join("vizualization_of_data_set", os.path.join(folder_name, 'merged_image' + str(n) + '.png')))
        new_image.save(path)

