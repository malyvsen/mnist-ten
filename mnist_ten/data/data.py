import torch
import torchvision as tv
from tqdm.auto import tqdm

from .config import data_path, num_classes, batch_size



train_full = tv.datasets.MNIST(data_path, download=True, train=True)
test = tv.datasets.MNIST(data_path, download=True, train=False)

train_labeled = {label: [] for label in range(num_classes)}
train_unlabeled = []
for image, label in tqdm(train_full):
    if len(train_labeled[label]) < 1:
        train_labeled[label].append(image)
    else:
        train_unlabeled.append(image)

train_labeled = [(image, label) for label, images in train_labeled.items() for image in images]

train_unlabeled_loader = torch.utils.data.DataLoader(train_unlabeled, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)