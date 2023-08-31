import torch
import lightning.pytorch as pl

from model import MyModel
from config import Config

#set device
device = torch.device()

#seed
pl.seed_everything(config.seed,workers=True)

#load config
config = Config().configargs

#define model
model = MyModel(**config)

#load checkpoint
dir = None
assert dir != None
model.load_from_checkpoint(dir,map_location=device)
model.eval()
######################testing######################
