import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision as tv

from . import mnist
from .rotoflip import rotoflip
from .augement import augment
from .transform_dataset import TransformDataset
from .config import batch_size



def collate_labeled(batch):
    images = [image for image, label in batch]
    labels = [label for image, label in batch]
    return default_collate(images), torch.tensor(labels)


train_rotoflips = TransformDataset(
    slave_dataset=[image for image, label in mnist.train_labeled] + mnist.train_unlabeled,
    transform=tv.transforms.Compose([
        tv.transforms.ToTensor(),
        rotoflip
    ])
)
train_labeled = TransformDataset(
    slave_dataset=mnist.train_labeled,
    transform=tv.transforms.Compose([
        augment,
        tv.transforms.ToTensor()
    ])
)
test = TransformDataset(
    slave_dataset=mnist.test,
    transform=tv.transforms.ToTensor()
)

train_rotoflip_loader = DataLoader(train_rotoflips, batch_size=batch_size, collate_fn=collate_labeled, shuffle=True)
train_labeled_loader = DataLoader(train_labeled, batch_size=batch_size, collate_fn=collate_labeled, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, collate_fn=collate_labeled, shuffle=False)