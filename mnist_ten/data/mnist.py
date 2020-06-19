import torchvision as tv

from .config import data_path, num_classes



train_full = tv.datasets.MNIST(data_path, download=True, train=True)
test = tv.datasets.MNIST(data_path, download=True, train=False)

train_labeled = {label: [] for label in range(num_classes)}
train_unlabeled = []
for image, label in train_full:
    if len(train_labeled[label]) < 1:
        train_labeled[label].append(image)
    else:
        train_unlabeled.append(image)

train_labeled = [(image, label) for label, images in train_labeled.items() for image in images]