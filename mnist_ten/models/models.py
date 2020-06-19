import numpy as np
import torch

from .squeeze_layer import SqueezeLayer
from mnist_ten.data import image_shape, num_classes
from . import config



shared = torch.nn.Sequential(
    torch.nn.Conv2d(image_shape[0], config.num_channels_between, kernel_size=3, padding=1),
    torch.nn.LeakyReLU(),
    *[SqueezeLayer(config.num_channels_between, config.num_channels_squeeze) for i in range(config.num_squeezes)],
    torch.nn.Conv2d(config.num_channels_between, config.num_channels_last, kernel_size=1),
    torch.nn.LeakyReLU(),
    torch.nn.Flatten(start_dim=1)
)

classifier_head = torch.nn.Sequential(
    torch.nn.Linear(config.flattened_length, 64),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(64, num_classes)
)

classifier = torch.nn.Sequential(
    shared,
    classifier_head
)

rotoflip_head = torch.nn.Sequential(
    torch.nn.Linear(config.flattened_length, 8)
)

rotoflip = torch.nn.Sequential(
    shared,
    rotoflip_head
)