from config import Config
from datasets import MyDataset,collate_fn
from model import MyModel

from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

#load config
config = Config().configargs

#seed
pl.seed_everything(config['seed'],workers=True)

#load data
dlconfig = config['dataloaderconfig']
test_data = MyDataset()
test_loader = DataLoader(test_data, collate_fn=collate_fn, **dlconfig)

#define model
model = MyModel(**config)
model.eval()
model.freeze()

#load checkpoint
ckpt = None

######################training######################
lr_monitor = LearningRateMonitor(logging_interval='step')
ckpt_monitor = ModelCheckpoint(verbose=True,every_n_train_steps = 1000,save_last =True, monitor='loss')
trainer = pl.Trainer(callbacks=[lr_monitor,ckpt_monitor],**config['trainerargs'])
trainer.test(model,dataloaders=test_loader,ckpt_path=ckpt)
