from config import Config
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from datasets import MyDataset,collate_fn
from model import MyModel

config = Config()
#seed
pl.seed_everything(config.seed,workers=True)
#load data
dlconfig = config.dataloaderconfig
training_data = MyDataset()
train_loader = DataLoader(training_data, collate_fn=collate_fn, **config.dataloaderconfig)

######################training######################
trainer = pl.Trainer(**config.trainerargs)
model = MyModel(config)
trainer.fit(model,train_dataloaders=train_loader)
