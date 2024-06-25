import sys,os
sys.path.append(os.getcwd())

from torch.utils.data import Dataset,DataLoader
import torch

def collate_fn(batch):
    batch = torch.as_tensor(batch)
    return batch

class MyDataset(Dataset):
    def __init__(self,data):
        self.data = data
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

##############TEST##############
if __name__ == '__main__':
    from tqdm import tqdm
    import yaml
    import lightning.pytorch as pl
    pl.seed_everything(42)
    #load config
    config = yaml.load(open("configs/default.yaml", "r"), Loader=yaml.FullLoader)
    trainingset = MyDataset()
    train_dataloader = DataLoader(trainingset, batch_size=1, num_workers=0, shuffle=True)
    for data in tqdm(train_dataloader):
        pass
