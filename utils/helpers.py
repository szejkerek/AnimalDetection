import numpy as np
import ssl
import os
from multiprocessing import freeze_support
import datetime
import torch

import config


def setup_env():
    freeze_support()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    ssl._create_default_https_context = ssl._create_unverified_context


def create_current_folder():
    root_dir = os.getcwd()
    results_dir = os.path.join(root_dir, "!Results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    current_time = datetime.datetime.now()
    subfolder_name = current_time.strftime("%d-%m_%H-%M-%S")
    subfolder_dir = os.path.join(results_dir, subfolder_name)
    if not os.path.exists(subfolder_dir):
        os.makedirs(subfolder_dir)
    return subfolder_dir


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def crop_to_divisible_by_32(image):
    h, w = image.shape[:2]
    h_new = h - h % 32
    w_new = w - w % 32
    cropped_image = image[:h_new, :w_new]
    return cropped_image


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def save_model():
    if config.IS_MODEL_SAVED:
        torch.save(config.model, os.path.join(config.CURRENT_PATH, "model.pth"))
        print('Model saved!')


def load_model():
    return torch.load(os.path.join(config.CURRENT_PATH, "model.pth"))


def continue_training(date="xd", enabled=False):
    if not enabled:
        return False

    root_dir = os.getcwd()
    results_dir = os.path.join(root_dir, "!Results")
    if not os.path.exists(results_dir):
        print("Results folder does not exists")
        return False

    subfolder_name = date
    subfolder_dir = os.path.join(results_dir, subfolder_name)
    if not os.path.exists(subfolder_dir):
        print("Subfolder: " + date + " does not exists")
        return False

    model_path = os.path.join(subfolder_dir, "model.pth")
    if not os.path.exists(model_path):
        print("Couldnt locate \"model.pth\" under path: " + model_path)
        return False

    config.model = torch.load(model_path)
    config.CURRENT_PATH = subfolder_dir

    print("##############################")
    print("#      RETRAINING MODEL      #")
    print("##############################")
