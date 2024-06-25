import yaml
import lightning.pytorch as pl
from datasets.dataset import MyDataset
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
import soundfile as sf

CKPTPATH = 'last.ckpt'
CONFIGPATH = 'configs/default.yaml'
TARGETPATH = 'testresults/'
DEVICE = 'cpu'
os.makedirs(TARGETPATH, exist_ok=True)

config = yaml.load(open(CONFIGPATH, "r"), Loader=yaml.FullLoader)
pl.seed_everything(config['seed'],workers=True)

#load data
dataconfig = config['data']
test_list = 
test_data = MyDataset(test_list)
config['dataloader']['batch_size'] = 1
config['dataloader']['num_workers'] = 0
test_loader = DataLoader(test_data, **config['dataloader'])

#load model
sds = torch.load(CKPTPATH, map_location=DEVICE)
sd, emad = sds['state_dict'], sds['ema']

model = model(config).to(DEVICE)
model.load_state_dict(sd)
model.ema.load_state_dict(emad)
model.eval()
model.freeze()
model.ema.copy_to()
with torch.inference_mode():
    for idx, data in enumerate(tqdm(test_loader)):    
        pass