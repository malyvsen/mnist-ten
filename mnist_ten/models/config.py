import numpy as np

from mnist_ten.data import image_shape



num_channels_between = 64
num_channels_squeeze = 32
num_channels_last = 4
flattened_length = np.prod(image_shape[1:]) * num_channels_last