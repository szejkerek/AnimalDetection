# Animal Detection in Camouflaged Environments

## Introduction

This project was conducted as a part of the Biologically Inspired Artificial Intelligence course at the Silesian University of Technology. The goal of our project was to develop an advanced neural network utilizing machine learning techniques to effectively detect animals hidden in their natural environments. To achieve this task, we utilized the PyTorch library and the Segmentation Models repository.

## Dataset

The dataset used for our research can be found in the `!Dataset\Images` directory of this repository. It consists of approximately 300 images featuring various animals in naturally camouflaging environments. During the data collection process, we took several aspects into account, including avoiding the central part of the frame and removing any overlaid text on the images. The dataset was prepared by the students involved in the project.

Below is an example image from the dataset along with its corresponding mask, which describes the presence of the animal in the image, the masking and non-masking backgrounds, and the attention-grabbing fragment.

![rjcc7hj0](https://github.com/szejkerek/AnimalDetection/assets/69083596/c554cad4-632b-4c40-9b5e-97ac2d764032)

*Example image with a camouflaged animal*

![ExampleMask](https://github.com/szejkerek/AnimalDetection/assets/69083596/370e7bfe-625a-4dfa-95e2-d81c7f405e6b)
*Example mask overlayed on the image*

## Network Training

The training process of our neural network was performed on an "NVIDIA 1070 GamingX" graphics card with 8 GB VRAM. We achieved efficient and satisfactory training by utilizing the CUDA Toolkit. During the training, we experimented with various network settings and parameters. The testing process was a combination of empirical error searching and exploring new issues and solutions. In the following sections, we will describe in detail the differences between the selected approaches.

## Program Output

In this section, we will provide a detailed description of the program's output, which was saved on the user's disk. Each training run saved multiple files, such as a file with learning statistics, a file with the neural network configuration, a file with the learning curve plot, and results from the test dataset. Below, we present these results with detailed explanations:

Example Learning Statistics - `stats.cfg`
The file contains information such as:
- Path to save statistics
- Duration of the training
- Number of epochs
- Space for any notes

Example Neural Network Configuration - `config.cfg`
The file contains information such as:
- Uused weights for each class
- Encoder
- Encoder weights
- Activation function
- Model
- Loss function
- Optimizer
- Learning rate
- Batch size

Example Learning Curve and Loss Plot:

![LearningCurve](https://github.com/szejkerek/AnimalDetection/assets/69083596/025ef893-559d-4263-8e5d-6b2b131d0a8d)

## Best Approach

After comparing different network settings, we have identified the best approach for our project. This approach yielded the most promising results and achieved a higher IoU score compared to the other configurations.

Approach:
- Mask weights: (1, 0.05, 0.03, 0.01)
- Model: UNet
- Encoder: Resnet34
- Batch size: 15
- Activation function: Softmax2d
- Loss function: CrossEntropy
- Optimizer: Adam
- Learning rate: 0.0001
- Encoder weights: ImageNet
- Sample result: IoU score of 0.51

By returning to the UNet architecture with pre-defined weights and utilizing a batch size of 15, we observed improved results. The network demonstrated less generalization, resulting in better animal detection performance. Based on these findings, we conclude that this approach provides the most optimal configuration for detecting animals in camouflaged environments.

Here are a few examples of output images obtained using the best approach:
![test_15](https://github.com/szejkerek/AnimalDetection/assets/69083596/6e22a4f4-0d41-4d06-982c-67ed42ae2935)

*Output image 1*
![test_8](https://github.com/szejkerek/AnimalDetection/assets/69083596/b5209eb6-a070-4975-b4cc-d56a96b5f964)

*Output image 2*
![test_13](https://github.com/szejkerek/AnimalDetection/assets/69083596/03aca162-8f50-4e75-96b8-b88a0a660467)

*Output image 3*
![test_9](https://github.com/szejkerek/AnimalDetection/assets/69083596/dfce35ef-4382-4959-84ae-a597eabdb7b1)

*Output image 4*
![test_5](https://github.com/szejkerek/AnimalDetection/assets/69083596/38d1caf9-c819-4a4d-ba78-7c28b6b7177a)

*Output image 5*
![test_0](https://github.com/szejkerek/AnimalDetection/assets/69083596/b3c14a20-b019-4290-bc41-4030c41f002d)

*Output image 6*
![test_7](https://github.com/szejkerek/AnimalDetection/assets/69083596/e88a3c14-3f38-4999-9bef-30fbf89f25d2)

*Output image 7*

## Run Locally

Follow these steps to run the project locally:

1. Clone the project repository

```bash
git clone https://github.com/szejkerek/AnimalDetection.git
```

2. Navigate to the project directory

```bash
cd AnimalDetection
```

3. Install the required dependencies from the `requirements.txt` file located in the root folder. Make sure you have Python and pip installed.

```bash
pip install -r requirements.txt
```

4. Install CUDA Toolkit (if not already installed) to utilize GPU acceleration. Refer to the official NVIDIA website for installation instructions specific to your operating system.

5. Install the latest version of PyTorch. You can visit the official PyTorch website for installation instructions based on your system configuration.

6. Run the `main.py` file to start the project

```bash
python main.py
```

## Conclusions

Based on our project, we have drawn several conclusions. First and foremost, we have confirmed that using neural networks and machine learning techniques can be an effective tool for detecting animals in camouflaged environments. The solution we developed achieved good results, but there is still potential for further improvement. Through this project, we gained valuable experience in the field of machine learning and neural networks, which we can leverage in future projects. We also learned that conducting successful neural network training does not require supercomputers and can be done in a home environment.

## License

This project utilizes the [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) library for advanced neural network architectures.

> Qubvel. segmentation_models.pytorch. Retrieved from: https://github.com/qubvel/segmentation_models.pytorch

## Code Highlights

### Color-Indexed Multi-Class Mask Extraction

```python
# data_model/dataset_class.py  (lines 66–73)
masks = []
for cls_value in self.class_values:
    color = config.COLORS[cls_value]
    newMask = cv2.inRange(mask, color, color)
    newMask = np.float32(newMask)
    newMask = cv2.normalize(newMask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    masks.append(newMask)
mask = np.stack(masks, axis=-1).astype('float32')
```

Each semantic class (animal, masking background, non-masking background, foreground attention) is encoded as a distinct RGB color in the ground-truth mask images. At load time, the Dataset iterates over the requested class list, performs exact-color range matching via `cv2.inRange`, normalizes each binary channel to [0, 1], and stacks them into a single `(H, W, C)` float tensor. This avoids storing separate mask files per class and cleanly decouples the segmentation task's multi-channel representation from the storage format.

### Stochastic Multi-Group Augmentation Pipeline

```python
# utils/augmentation.py  (lines 4–37)
def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.1, p=0.9, border_mode=0),
        albu.PadIfNeeded(min_height=544, min_width=544, always_apply=True, border_mode=0),
        albu.RandomCrop(height=544, width=544, always_apply=True),
        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=0.5),
                albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                albu.RandomGamma(gamma_limit=(80, 120), p=0.5),
                albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            ],
            p=0.8,
        ),
        albu.OneOf(
            [
                albu.GaussianBlur(blur_limit=(3, 7), p=0.5),
                albu.MedianBlur(blur_limit=3, p=0.5),
                albu.MotionBlur(blur_limit=(3, 7), p=0.5),
            ],
            p=0.5,
        ),
        albu.OneOf(
            [
                albu.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                albu.ImageCompression(quality_lower=85, quality_upper=95, p=0.5),
            ],
            p=0.5,
        ),
    ]
    return albu.Compose(train_transform)
```

The augmentation pipeline is structured into three independently-sampled `OneOf` groups targeting distinct types of photometric variation: tone/color (CLAHE, brightness/contrast, gamma, HSV shift), blur type (Gaussian, median, motion), and signal degradation (noise, JPEG compression). This grouping reflects deliberate reasoning about the visual properties of camouflage — animals in natural settings are subject to lighting variation, camera blur, and image quality differences — and prevents augmentations from the same perceptual category from compounding on a single sample.
