import tensorflow as tf
from tensorflow import keras
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.debugging.set_log_device_placement(True)

print(tf.config.experimental.list_physical_devices())
print(tf.__version__)

print(tf.config.list_physical_devices("GPU"))
