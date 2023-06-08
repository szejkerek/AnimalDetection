import time
import keyboard as keyboard
import torch

import config
import data_model
import utils
from utils import save_results

if not utils.continue_training(enabled=False):
    print("Starting new training routine...")
    config.CURRENT_PATH = utils.create_current_folder()

utils.setup_env()

# utils.save_visualization(data_model.train_visualize, "Train", enabled=True)
# utils.save_visualization(data_model.valid_visualize, "Valid", enabled=True)
# utils.save_visualization(data_model.test_visualize, "Test", enabled=True)

max_score = 0

start_time = time.time()
while True:
    torch.cuda.empty_cache()
    # Key must be hold at the start of loop cycle
    if keyboard.is_pressed(config.INTERRUPT_KEY):
        print("Learning interrupted by user.")
        break

    print('\nEpoch: {}'.format(config.EPOCH_COUNT))

    train_logs = data_model.train_epoch.run(data_model.train_loader)
    valid_logs = data_model.valid_epoch.run(data_model.valid_loader)

    if config.EPOCH_COUNT % 30 == 0:
        test_log = data_model.evaluate_test_data()

    test_log = 0

    learning_score = valid_logs['iou_score']

    utils.update_plot(train_logs, valid_logs, test_log, enabled=True)

    if max_score < learning_score:
        max_score = learning_score
        utils.save_model()

    if config.EPOCH_COUNT == 30:
        config.optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

    if config.EPOCH_COUNT == 200:
        config.optimizer.param_groups[0]['lr'] = 1e-6
        print('Decrease decoder learning rate to 1e-6!')

    config.EPOCH_COUNT += 1

config.ELAPSED_TIME = time.time() - start_time
test_log = data_model.evaluate_test_data()
config.save_stats(test_log)
save_results(data_model.test_visualize, data_model.test_dataset, count=-1)
