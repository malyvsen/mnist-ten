import numpy as np
import torch

from .squeeze_layer import SqueezeLayer
from mnist_ten.data import image_shape, num_classes
from .config import num_channels_between, num_channels_squeeze, num_channels_last, flattened_length



common = torch.nn.Sequential(
    torch.nn.Conv2d(image_shape[0], num_channels_between, kernel_size=3, padding=1),
    torch.nn.LeakyReLU(),
    *[SqueezeLayer(num_channels_between, num_channels_squeeze) for i in range(3)],
    torch.nn.Conv2d(num_channels_between, num_channels_last, kernel_size=1),
    torch.nn.LeakyReLU(),
    torch.nn.Flatten(start_dim=1)
)

classifier = torch.nn.Sequential(
    common,
    torch.nn.Linear(flattened_length, 64),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(64, num_classes)
)

rotoflip_classifier = torch.nn.Sequential(
    common,
    torch.nn.Linear(flattened_length, 64),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(64, 8)
)