from torch.utils.data import Dataset,DataLoader
import torch

from config import DataLoaderConfig

class XXXDataset(Dataset):
    def __init__(self,data):
        self.data = data
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

##############TEST##############
'''
config = DataLoaderConfig()
training_data = XXXDataset(data)
train_dataloader = DataLoader(training_data, batch_size=config.batch_size,shuffle=config.shuffle,num_workers=config.num_workers,pin_memory=config.pin_memory,collate_fn=config.collate_fn)
train_data_iter = next(iter(train_dataloader))
print(train_data_iter)
'''