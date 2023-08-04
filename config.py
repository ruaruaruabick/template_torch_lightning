import numpy as np
import torch
class DataLoaderConfig(object):
    batch_size = 64
    shuffle = True
    num_workers = 16
    pin_memory = True
    drop_last = False

    def collate_fn(self, batch):
        batch = torch.as_tensor(batch)
        return batch