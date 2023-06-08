import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
import config
import utils
from utils import get_non_repeating_numbers


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
    utils.create_subfolder_in_date_folder("vizualization_of_data_set\\" + folder_name)
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

        # Add red border of 1px to each image
        normal_image = ImageOps.expand(normal_image, border=1, fill='red')
        animals = ImageOps.expand(animals, border=1, fill='red')
        masking_background = ImageOps.expand(masking_background, border=1, fill='red')
        nonmasking_background = ImageOps.expand(nonmasking_background, border=1, fill='red')
        foreground_attention = ImageOps.expand(foreground_attention, border=1, fill='red')

        new_image.paste(normal_image, (0, 0))

        new_image.paste(animals, (normal_image.width, 0))
        new_image.paste(foreground_attention, (int(normal_image.width + normal_image.width / 2), 0))

        new_image.paste(masking_background, (normal_image.width, int(normal_image.height / 2)))
        new_image.paste(nonmasking_background,
                        (int(normal_image.width + normal_image.width / 2), int(normal_image.height / 2)))

        path = os.path.join(config.CURRENT_PATH, os.path.join("vizualization_of_data_set", os.path.join(folder_name,
                                                                                                        'merged_image' + str(
                                                                                                            n) + '.png')))
        new_image.save(path)


train_scores = []
valid_scores = []
test_scores = []

train_losses = []
valid_losses = []
test_losses = []

first_launch = True
fig_number = 0


def update_plot(train_logs, valid_logs, test_log, enabled=True):
    if not enabled:
        return False

    if plt.get_fignums():
        fig = plt.figure(1, figsize=(16, 8), dpi=80)
        fig.clf()
        axs = fig.subplots(1, 2)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=80)

    train_score = config.calculate_score(train_logs)
    valid_score = config.calculate_score(valid_logs)
    test_score = 0  # config.calculate_score(test_log)

    train_loss = config.calculate_loss(train_logs)
    valid_loss = config.calculate_loss(valid_logs)
    test_loss = 0  # config.calculate_loss(test_log)

    train_scores.append(train_score)
    valid_scores.append(valid_score)
    test_scores.append(test_score)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    test_losses.append(test_loss)

    axs[0].plot(train_scores, color='blue', label='Train score')
    axs[0].plot(valid_scores, color='orange', label='Valid score')
    axs[0].plot(test_scores, color='red', label='Train score')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Score')
    axs[0].legend(loc='upper left')
    axs[0].set_title("Train score: {}\nValid score: {}\nTest score: {}".format(round(train_score, 6),
                                                                               round(valid_score, 6),
                                                                               round(test_score, 6)))

    axs[1].plot(train_losses, color='blue', label='Train loss')
    axs[1].plot(valid_losses, color='orange', label='Valid loss')
    axs[1].plot(test_losses, color='red', label='Train loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc='upper right')
    axs[1].set_title("Train loss: {}\nValid loss: {}\nTest loss: {}".format(round(train_loss, 3),
                                                                            round(valid_loss, 3),
                                                                            round(test_loss, 3)))

    fig.canvas.flush_events()
    plt.pause(5)
    plt.tight_layout()
    plt.savefig(os.path.join(config.CURRENT_PATH, "fig.png"))


def save_results(test_visualize, test_dataset, count=10):
    utils.create_subfolder_in_date_folder("test_result")
    for i in get_non_repeating_numbers(len(test_dataset), count):
        # Extract gt and pr masks from test_model
        image_vis = test_visualize[i][0].astype('uint8')
        image, gt_mask = test_dataset[i]
        filename = test_dataset.get_name(i)
        print("Saving "+filename+'...')

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(config.DEVICE).unsqueeze(0)
        pr_mask = config.model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        # Convert images and masks to PTL images
        normal_image = Image.fromarray(np.uint8(image_vis)).resize((512, 512))

        gt_animals = Image.fromarray(np.uint8(gt_mask[0, ...].squeeze() * 255)).convert('RGB')
        gt_masking_background = Image.fromarray(np.uint8(gt_mask[1, ...].squeeze() * 255)).convert('RGB')
        gt_nonmasking_background = Image.fromarray(np.uint8(gt_mask[2, ...].squeeze() * 255)).convert('RGB')
        gt_foreground_attention = Image.fromarray(np.uint8(gt_mask[3, ...].squeeze() * 255)).convert('RGB')

        pr_animals = Image.fromarray(np.uint8(pr_mask[0, ...].squeeze() * 255)).convert('RGB')
        pr_masking_background = Image.fromarray(np.uint8(pr_mask[1, ...].squeeze() * 255)).convert('RGB')
        pr_nonmasking_background = Image.fromarray(np.uint8(pr_mask[2, ...].squeeze() * 255)).convert('RGB')
        pr_foreground_attention = Image.fromarray(np.uint8(pr_mask[3, ...].squeeze() * 255)).convert('RGB')

        # Create canvas and resize default image
       #normal_image = normal_image.resize((normal_image.width * 2, normal_image.height * 2))
        normal_image = normal_image.resize((normal_image.width * 2, normal_image.height * 2))
        new_image = Image.new('RGB', (normal_image.width * 3, normal_image.height))

        block_width = int(normal_image.width / 2)
        block_height = int(normal_image.height / 2)
        new_image.paste(normal_image, (0, 0))

        # Add red border to each image
        gt_animals = ImageOps.expand(gt_animals, border=1, fill='red')
        gt_masking_background = ImageOps.expand(gt_masking_background, border=1, fill='red')
        gt_nonmasking_background = ImageOps.expand(gt_nonmasking_background, border=1, fill='red')
        gt_foreground_attention = ImageOps.expand(gt_foreground_attention, border=1, fill='red')
        pr_animals = ImageOps.expand(pr_animals, border=1, fill='red')
        pr_masking_background = ImageOps.expand(pr_masking_background, border=1, fill='red')
        pr_nonmasking_background = ImageOps.expand(pr_nonmasking_background, border=1, fill='red')
        pr_foreground_attention = ImageOps.expand(pr_foreground_attention, border=1, fill='red')

        # Paste images on canvas to be merged
        new_image.paste(gt_animals, (block_width * 2, 0))
        new_image.paste(gt_masking_background, (block_width * 3, 0))
        new_image.paste(gt_nonmasking_background, (block_width * 4, 0))
        new_image.paste(gt_foreground_attention, (block_width * 5, 0))

        new_image.paste(pr_animals, (block_width * 2, block_height))
        new_image.paste(pr_masking_background, (block_width * 3, block_height))
        new_image.paste(pr_nonmasking_background, (block_width * 4, block_height))
        new_image.paste(pr_foreground_attention, (block_width * 5, block_height))

        # Save file
        path = os.path.join(config.CURRENT_PATH,
                            os.path.join("test_result", filename + '_result' + '.png'))
        new_image.save(path)
