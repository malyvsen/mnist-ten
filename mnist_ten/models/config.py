import os
import numpy as np

from mnist_ten.data import image_shape



weights_path = os.path.join(os.path.dirname(__file__), 'classifier_weights.pth')

num_squeezes = 2
num_channels_between = 64
num_channels_squeeze = 32
num_channels_last = 64
flattened_length = np.prod(image_shape[1:]) * num_channels_last // (4 ** num_squeezes)