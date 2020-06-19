import numpy as np
import torch



class TransformDataset(torch.utils.data.Dataset):
    '''
    Transforms a labeled dataset using the provided callable transform.
    '''
    def __init__(self, slave_dataset, transform):
        super().__init__()
        self.slave = slave_dataset
        self.transform = transform
    

    def __len__(self):
        return len(self.slave)
    

    def __getitem__(self, idx):
        sample = self.slave[idx]
        if isinstance(sample, tuple):
            return self.transform(sample[0]), *sample[1:]
        return self.transform(sample)