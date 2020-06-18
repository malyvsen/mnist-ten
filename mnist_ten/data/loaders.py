import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate 

from .mnist import train_labeled, train_unlabeled, test
from .rotoflip_dataset import RotoflipDataset
from .config import batch_size


def collate_labeled(batch):
    images = [image for image, label in batch]
    labels = [label for image, label in batch]
    return default_collate(images), torch.tensor(labels)



train_rotoflips = RotoflipDataset([image for image, label in train_labeled] + train_unlabeled)

train_rotoflip_loader = DataLoader(train_rotoflips, batch_size=batch_size, collate_fn=collate_labeled, shuffle=True)
train_labeled_loader = DataLoader(train_labeled, batch_size=batch_size, collate_fn=collate_labeled, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, collate_fn=collate_labeled, shuffle=False)