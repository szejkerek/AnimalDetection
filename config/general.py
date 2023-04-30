import os

EPOCH_COUNT = 0
CURRENT_PATH = ""
IS_MODEL_SAVED = False

CLASSES = ['animal', 'maskingbackground', 'nonmaskingbackground', 'nonmaskingforegroundattention']
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]

DATA_DIR = './!Dataset'

x_train_dir = os.path.join(DATA_DIR, 'GDTrain/Images')
y_train_dir = os.path.join(DATA_DIR, 'GDTrain/Masks')

x_valid_dir = os.path.join(DATA_DIR, 'GDValidation/Images')
y_valid_dir = os.path.join(DATA_DIR, 'GDValidation/Masks')

x_test_dir = os.path.join(DATA_DIR, 'GDTest/Images')
y_test_dir = os.path.join(DATA_DIR, 'GDTest/Masks')

DEVICE = 'cuda'

INTERRUPT_KEY = 'esc'
ELAPSED_TIME = 0
