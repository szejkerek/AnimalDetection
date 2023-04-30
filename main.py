import datetime
import time
import keyboard as keyboard
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
    # Key must be hold at the start of loop cycle
    if keyboard.is_pressed(config.INTERRUPT_KEY):
        print("Learning interrupted by user.")
        break

    print('\nEpoch: {}'.format(config.EPOCH_COUNT))

    train_logs = data_model.train_epoch.run(data_model.train_loader)
    valid_logs = data_model.valid_epoch.run(data_model.valid_loader)
    test_log = data_model.evaluate_test_data()

    learning_score = config.calculate_score(valid_logs)

    utils.update_plot(train_logs, valid_logs, test_log, enabled=False)

    if max_score < learning_score:
        max_score = learning_score
        utils.save_model()

    if config.EPOCH_COUNT == 25:
        config.optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
    config.EPOCH_COUNT += 1

end_time = time.time()
config.ELAPSED_TIME = end_time - start_time
formatted_time = str(datetime.timedelta(seconds=config.ELAPSED_TIME)).split(".")[0]
print("Time of learning", formatted_time)

save_results(data_model.test_visualize, data_model.test_dataset, count=1)