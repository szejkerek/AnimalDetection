import os

EPOCH_COUNT = 0
CURRENT_PATH = ""
IS_MODEL_SAVED = False

CLASSES = ['animal', 'maskingbackground', 'nonmaskingbackground', 'nonmaskingforegroundattention']
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]

DATA_DIR = './!Dataset/Dataset1'

x_train_dir = os.path.join(DATA_DIR, 'Train/Images')
y_train_dir = os.path.join(DATA_DIR, 'Train/Masks')

x_valid_dir = os.path.join(DATA_DIR, 'Validation/Images')
y_valid_dir = os.path.join(DATA_DIR, 'Validation/Masks')

x_test_dir = os.path.join(DATA_DIR, 'Test/Images')
y_test_dir = os.path.join(DATA_DIR, 'Test/Masks')

DEVICE = 'cuda'

INTERRUPT_KEY = 'esc'
ELAPSED_TIME = 0
