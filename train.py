import yaml

from datasets import MyDataset,collate_fn
from model import MyModel

from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

#for high-performance gpu
torch.set_float32_matmul_precision('high')

#load config
config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

#seed
pl.seed_everything(config['seed'],workers=True)

#load data
training_data = MyDataset()
train_loader = DataLoader(training_data, collate_fn=collate_fn, **config['dataloader'])

#define model
model = MyModel(**config)

#load checkpoint
ckpt = None

######################training######################
lr_monitor = LearningRateMonitor(logging_interval='step')
ckpt_monitor = ModelCheckpoint(verbose=True,every_n_train_steps = 1000,save_last =True, save_top_k=-1, dirpath=None)
trainer = pl.Trainer(callbacks=[lr_monitor,ckpt_monitor],**config['trainerargs'])
trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=valid_loader,ckpt_path=ckpt)
