import datetime
import os

import config
from utils import config_line

ELAPSED_TIME = 0
EPOCH_COUNT = 0
CURRENT_PATH = ""


def save_stats():
    path = config.CURRENT_PATH
    elapsed = config.ELAPSED_TIME
    epoch = config.EPOCH_COUNT
    print("Saving stats...")
    f = open(os.path.join(path, "stats.cfg"), "w")
    formatted_time = str(datetime.timedelta(seconds=elapsed)).split(".")[0]
    f.write(config_line("InitialPath", path))
    f.write(config_line("ElapsedTime", formatted_time))
    f.write(config_line("EpochCount", epoch))
    f.write(config_line("Notes", "\"\n\n\""))

    f.close()
