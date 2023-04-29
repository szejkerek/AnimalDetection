import torch
import data_model
import config
import utils

if not utils.continue_training(enabled=False):
    print("Starting new training routine...")
    config.CURRENT_PATH = utils.create_current_folder()

utils.setup_env()

#utils.save_visualization(data_model.train_visualize, enabled=True)
#utils.save_visualization(data_model.valid_visualize, enabled=True)
#utils.save_visualization(data_model.test_visualize, enabled=True)

max_score = 0
for i in range(0, 10):
    print('\nEpoch: {}'.format(i))
    train_logs = data_model.train_epoch.run(data_model.train_loader)
    valid_logs = data_model.valid_epoch.run(data_model.valid_loader)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        utils.save_model()

    if i == 25:
        config.optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

test_log = data_model.evaluate_test_data()

for i in range(0):
    image_vis = data_model.test_visualize[i][0].astype('uint8')
    image, gt_mask = data_model.test_dataset[i]

    gt_mask = gt_mask.squeeze()

    x_tensor = torch.from_numpy(image).to(config.DEVICE).unsqueeze(0)
    pr_mask = config.model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    utils.visualize(
        image=image_vis,
        ground_truth_mask=gt_mask[0, ...].squeeze(),
        animal=pr_mask[0, ...].squeeze(),
        masking=pr_mask[1, ...].squeeze(),
        nonmasking=pr_mask[2, ...].squeeze(),
        attention=pr_mask[3, ...].squeeze(),
    )