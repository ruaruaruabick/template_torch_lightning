import yaml
import os
import argparse

from datasets import MyDataset,collate_fn
from model import MyModel

from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

assert torch.cuda.is_available(), "CPU training is not allowed."
#for high-performance gpu
torch.set_float32_matmul_precision('high')

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
    ckpt = "lightning_logs/{}.ckpt".format(args.restore_step)
    ckpt = ckpt if os.path.exists(ckpt) else None
    if args.finetune:
        model = Model.load_from_checkpoint(ckpt,config = config)
        ckpt = None
    else:
        model = Model(config)
        
    ######################training######################
    lr_monitor = LearningRateMonitor(logging_interval='step')
    ckpt_monitor = ModelCheckpoint(verbose=True,every_n_train_steps = 1000,save_last =True, save_top_k=-1, dirpath=args.output_dir)
    trainer = pl.Trainer(callbacks=[lr_monitor,ckpt_monitor],**config['trainerargs'])
    trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=valid_loader,ckpt_path=ckpt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
        type=str,
        default="config.yaml",
        help="config file path",
    )
    parser.add_argument("--restore_step",
        type=str,
        default="last",
        help="restore_step of ckpt",
    )
    parser.add_argument('--finetune', '-ft',
                        action='store_true',
                        default=False,
                        help='if finetune step, true means load base model'
    )
    parser.add_argument('--output_dir',
                        type=str,
                        default="lightning_logs/"
                        help='Directory to save checkpoints',
    )
    args = parser.parse_args()
    main(args)