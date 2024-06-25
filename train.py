import yaml
import os
import argparse
import glob

from datasets.dataset import MyDataset,collate_fn
from model import MyModel

from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
from lightning.pytorch.loggers import TensorBoardLogger

assert torch.cuda.is_available(), "CPU training is not allowed."
#for high-performance gpu
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main(args):
    #load config
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    #seed
    pl.seed_everything(config['seed'],workers=True)

    #load data
    training_data = MyDataset()
    train_loader = DataLoader(training_data, collate_fn=collate_fn, **config['dataloader'])

    #define model
    model = MyModel(config)

    #load checkpoint
    ckpt_dir = os.path.join(args.output_dir,args.expname+'_ckpt')
    if args.restore_step != "":
        pattern = f'{ckpt_dir}/epoch=*-step={args.restore_step}.ckpt'
    else:
        pattern = f'{ckpt_dir}/epoch=*-step=*.ckpt'
    ckpts = glob.glob(pattern)
        
    if len(ckpts) == 0:
        ckpt = None
    else:
        ckpt = sorted(ckpts,reverse=True, key=lambda x:int(x.split('=')[-1].split('.')[0]))[0]
        ckpt = ckpt if os.path.exists(ckpt) else None
        
    #load ema
    if ckpt is not None:
        sd = torch.load(ckpt, map_location='cpu')
        if 'ema' in sd:
            model.ema.load_state_dict(sd['ema'])
    
    ######################training######################
    lr_monitor = LearningRateMonitor(logging_interval='step')
    ckpt_monitor = ModelCheckpoint(verbose=True,every_n_train_steps=config['save_every_n_steps'],save_last =True, save_top_k=-1, dirpath=ckpt_dir)
    logger = TensorBoardLogger(save_dir=args.output_dir, name='lightning_logs',version=args.expname,default_hp_metric=False)
    trainer = pl.Trainer(callbacks=[lr_monitor,ckpt_monitor], logger=logger, profiler="simple", **config['trainer']) 
    trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=valid_loader,ckpt_path=ckpt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
        type=str,
        default="configs/default.yaml",
        help="config file path",
    )
    parser.add_argument("--expname",
        type=str,
        default="test",
        help="config file path",
    )
    parser.add_argument("--restore_step",
        type=str,
        default="",
        help="restore_step of ckpt",
    )
    parser.add_argument('--finetune', '-ft',
                        action='store_true',
                        default=False,
                        help='if finetune step, true means load base model'
    )
    parser.add_argument('--output_dir',
                        type=str,
                        default="lightning_logs/",
                        help='Directory to save checkpoints',
    )
    args = parser.parse_args()
    main(args)
