from config import Config
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from datasets import MyDataset,collate_fn
from model import MyModel

#load config
config = Config().configargs

#seed
pl.seed_everything(config['seed'],workers=True)

#load data
dlconfig = config['dataloaderconfig']
training_data = MyDataset()
train_loader = DataLoader(training_data, collate_fn=collate_fn, **dlconfig)

#define model
model = MyModel(**config)

#load checkpoint
ckpt = None
######################training######################
trainer = pl.Trainer(**config['trainerargs'])
trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=valid_loader,ckpt_path=ckpt)
