from torch.utils.data import DataLoader
import torch
import ssl
import os
from multiprocessing import freeze_support

import data_model
import config
import utils

freeze_support()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context

for i in range(1):
    image, mask = data_model.train_visualize[i]  # get some sample
    utils.visualize(
        image=image,
        animals=mask[..., 0].squeeze(),
        maskingbackground=mask[..., 1].squeeze(),
        nonmaskingbackground=mask[..., 2].squeeze(),
        nonmaskingforegroundattention=mask[..., 3].squeeze(),
    )

# train model for 40 epochs

max_score = 0

for i in range(0, 50):

    print('\nEpoch: {}'.format(i))
    train_logs = data_model.train_epoch.run(data_model.train_loader)
    valid_logs = data_model.valid_epoch.run(data_model.valid_loader)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(config.model, './best_model.pth')
        print('Model saved!')

    if i == 25:
        config.optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

# evaluate model on test data
test_logs = data_model.test_epoch.run(data_model.test_dataloader)

for i in range(3):
    image_vis = data_model.test_visualize[i][0].astype('uint8')
    image, gt_mask = data_model.test_dataset[i]

    gt_mask = gt_mask.squeeze()

    x_tensor = torch.from_numpy(image).to(config.DEVICE).unsqueeze(0)
    pr_mask = config.get_best_model().predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    utils.visualize(
        image=image_vis,
        ground_truth_mask=gt_mask[0, ...].squeeze(),
        animal=pr_mask[0, ...].squeeze(),
        masking=pr_mask[1, ...].squeeze(),
        nonmasking=pr_mask[2, ...].squeeze(),
        attention=pr_mask[3, ...].squeeze(),
    )
