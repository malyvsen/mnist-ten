from torch.utils.data import DataLoader

from .mnist import train_labeled, train_unlabeled, test
from .rotoflip_dataset import RotoflipDataset
from .config import batch_size



train_rotoflips = RotoflipDataset([image for image, label in train_labeled] + train_unlabeled)

train_rotoflip_loader = DataLoader(train_rotoflips, batch_size=batch_size, shuffle=True)
train_labeled_loader = DataLoader(train_labeled, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)